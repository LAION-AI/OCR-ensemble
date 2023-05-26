import sys, os, cv2, torch, argparse, ray, copy
sys.path.append(os.path.abspath('..'))
from tqdm import tqdm
import numpy as np
from functools import partial
from collections import defaultdict
from copy import deepcopy
from ray import serve
from torch.utils.data import DataLoader
from ocr_ensemble.classifiers import ClipEmbedding, ClipMulticlass, ClipPresence
from ocr_ensemble.proposers import PaddleOCRProposalGenerator
from ocr_ensemble.proposers import rotatedCrop
from ocr_ensemble.experts import HandwrittenExpert, Stage1Expert, PaddleOCRExpert
import tempfile
from typing import List
from torch.utils.data import default_collate
from ocr_ensemble.data import load_dataset

@ray.remote
class ClipPresenceRemote:
    def __init__(self, clip_presence_pickle, device='cuda'):
        self.device = device
        self.clip_emb = ClipEmbedding()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(clip_presence_pickle)
            self.temp_path = tmp.name
        self.clip_presence = ClipPresence(clip_emb=self.clip_emb, model_directory=self.temp_path)
        
    async def __call__(self, batch):
        return await self.process_batch(batch)
    
    def process_batch(self, batch):
        dataset_clip = deepcopy(batch)
        x_tuple, y_tuple, z_tuple = dataset_clip
        x_tuple_transformed = tuple(self.clip_presence.get_transform()(item) for item in x_tuple)
        dataset_clip = list(zip(x_tuple_transformed, y_tuple, z_tuple))
        features, labels, keys, label_dict  = [], [], [], {}

        for imgs, captions, key in tqdm(dataset_clip, desc="Calculating CLIP presence"):
            imgs = imgs.unsqueeze(0)
            preds = self.clip_presence.predict(imgs.to(self.device))
            labels += [preds]
            keys += [key]

        for key, pred in zip(keys, labels):
            label_dict[key] = pred

        labels = np.concatenate(labels, axis=0)
        return features, labels, label_dict

@ray.remote
class PaddleProposalRemote:
    def __init__(self):
        self.proposer = PaddleOCRProposalGenerator(device='cpu')

    def get_crops(self, src, label_dict, proposer):
        img, caption, key = src
        img = img.cpu().numpy()
        if label_dict[key] == 1:
            crops, bboxes = proposer(img)
            for crop, bbox in zip(crops, bboxes):
                yield crop, bbox, key
        if label_dict[key] == 0:
            yield img, None, key

    async def __call__(self, batch, label_dict):
        return await self.process_batch(batch, label_dict)

    def process_batch(self, batch, label_dict):
        dataset_crops = deepcopy(batch)
        x_tuple, y_tuple, z_tuple = dataset_crops
        dataset_crops = (x_tuple, y_tuple, z_tuple)
        dataset_crops = list(zip(x_tuple, y_tuple, z_tuple))
        result = [list(self.get_crops(src, label_dict, self.proposer)) for src in dataset_crops]
        return result
    
@ray.remote
class ClipMulticlassRemote:
    def __init__(self, moe_clf, moe_labels, device="cuda", expert_size="large"):  
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(moe_clf)
            self.temp_clf = tmp.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(moe_labels)
            self.temp_labels = tmp.name

        self.device = device  
        self.clip_emb = ClipEmbedding()
        self.expert_text_dict = {f"trocr-{expert_size}-handwritten": "handwritten text, handwriting, black on white",
                    "paddleocr": "text in a document, website, or presentation",
                    f"trocr-{expert_size}-stage1": "text in a scene"}
        self.clf = ClipMulticlass(list(self.expert_text_dict.values()),
                         clip_emb = self.clip_emb, model_directory=self.temp_clf, 
                         targets_directory=self.temp_labels)
    
    async def __call__(self, batch):
        return await self.process_batch(batch)
    
    def process_batch(self, batch):
        batch = deepcopy(batch)
        processed = []

        for minibatch in batch:
            for datapoint in minibatch:
                x_tuple, y_tuple, z_tuple = datapoint
                if y_tuple == None:
                    continue
                x_tuple_transformed = self.clip_emb.get_transform()(x_tuple)
                processed.append((x_tuple_transformed, y_tuple, z_tuple))
        

        bbox_dict = defaultdict(list)
        bbox_label_dict = defaultdict(list)
        bbox_scores_dict = defaultdict(list)

        for crop, bbox, key in tqdm(processed):
            crop = crop.unsqueeze(0)
            label = self.clf.predict(crop.to(self.device))
            bbox_dict[key] += [bbox]
            bbox_label_dict[key] += [label]

        return bbox_dict, bbox_label_dict
    
@ray.remote
class TrOcrRemote:
    def __init__(self, expert_size):
        self.expert = HandwrittenExpert(expert_size)

    async def __call__(self, crops):
        return await self.process_batch(crops)
    
    def process_batch(self, crops):
        return self.expert.process_batch(crops)
    
    def get_transform(self, x):
        return self.expert.get_transform()(x)
    
@ray.remote
class PaddleOCRRemote:
    def __init__(self, expert_size):
        self.expert = PaddleOCRExpert(device='cpu')
    
    async def __call__(self, crops):
        return await self.process_batch(crops)
    
    def process_batch(self, crops):
        return self.expert.process_batch(crops)
    
    def get_transform(self, x):
        return self.expert.get_transform()(x)
    
@ray.remote
class TrOcrStage1Remote:
    def __init__(self, expert_size):
        self.expert = Stage1Expert(expert_size)

    async def __call__(self, crops):
        return await self.process_batch(crops)
    
    def process_batch(self, crops):
        return self.expert.process_batch(crops)
    
    def get_transform(self, x):
        return self.expert.get_transform()(x)

@serve.deployment(route_prefix="/process_batch", name="HeadBatchProcessor", num_replicas=1)
class HeadBatchProcessor:
    def __init__(self, presence_clf, moe_clf, moe_labels, device='cuda', expert_size='large'):
        self.device = device
        self.expert_text_dict = {f"trocr-{expert_size}-handwritten": "handwritten text, handwriting, black on white",
                    "paddleocr": "text in a document, website, or presentation",
                    f"trocr-{expert_size}-stage1": "text in a scene"}
        
        self.clip_presence = ClipPresenceRemote.remote(presence_clf)
        self.proposer = PaddleProposalRemote.remote()
        self.clip_multiclass = ClipMulticlassRemote.options(name="ClipMulticlassRemote").remote(moe_clf, moe_labels)
        self.TrOcrRemote = TrOcrRemote.remote(expert_size)
        self.PaddleOCRRemote = PaddleOCRRemote.remote(expert_size)
        self.TrOcrStage1Remote = TrOcrStage1Remote.remote(expert_size)

        self.expert_mapping = {f"trocr-{expert_size}-handwritten": self.TrOcrRemote,
                                "paddleocr": self.PaddleOCRRemote,
                                f"trocr-{expert_size}-stage1": self.TrOcrStage1Remote}

    def get_efficient_and_filtered_crops(self, src, expert_idx, bbox_dict, bbox_label_dict):
        if(len(src) == 3):
            img, caption, key = src
            img = img.cpu().numpy()
            bboxes = bbox_dict[key]
            labels = bbox_label_dict[key]
            for bbox_idx, (bbox, label) in enumerate(zip(bboxes, labels)):
                if int(label[0]) == int(expert_idx):
                    crop = rotatedCrop(img, bbox)
                    yield crop, bbox_idx, key
        else:
            return
    
    def process_batch(self, batch):
        future_embedding = self.clip_presence.process_batch.remote(batch)
        features, labels, label_dict  = ray.get(future_embedding)

        future_proposal = self.proposer.process_batch.remote(batch, label_dict)
        batch_crops = ray.get(future_proposal)

        future_multiclass = self.clip_multiclass.process_batch.remote(batch_crops)
        bbox_dict, bbox_label_dict = ray.get(future_multiclass)

        ocr_dict = defaultdict(list)
        future_results = []

        batch_copy = deepcopy(batch)
        x_tuple, y_tuple, z_tuple = batch_copy
        batch_copy = list(zip(x_tuple, y_tuple, z_tuple))

        for idx, key in enumerate(self.expert_text_dict.keys()):
            expert = self.expert_mapping[key]
            batch_filtered = [list(self.get_efficient_and_filtered_crops(src, idx, bbox_dict, bbox_label_dict)) for src in batch_copy]
            processed = []
            batch_filtered = [x for x in batch_filtered if x != []]
            for minibatch in batch_filtered:
                for x_tuple, y_tuple, z_tuple in minibatch: 
                    x_tuple = torch.from_numpy(x_tuple).to(self.device)
                    x_tuple_transformed = ray.get(expert.get_transform.remote(x_tuple))
                    processed.append((x_tuple_transformed, y_tuple, z_tuple))
            
            for crops, bbox_ids, keys in tqdm(processed):
                if crops != None:
                    if key == 'paddleocr':
                        crops = crops.cpu().numpy()
                    else:
                        crops = crops.unsqueeze(0)
                    future_results.append((expert.process_batch.remote(crops), bbox_ids, keys))
        
        for future, bbox_id, key in future_results:
            text = ray.get(future)
            ocr_dict[key] += [(bbox_id, text)]

        ocr_dict_sorted = defaultdict(list)
        for key in ocr_dict.keys():
            list_of_tuples = ocr_dict[key]
            sorted_list_of_tuples = sorted(list_of_tuples, key=lambda x: x[0])
            ocr_dict_sorted[key] = [t[1] for t in sorted_list_of_tuples]

        expert_name_dict = defaultdict(list)
        expert_names = list(self.expert_text_dict.keys())
        for key, val in bbox_label_dict.items():
            expert_name_dict[key] = [expert_names[int(idx[0])] for idx in val]

        print(ocr_dict_sorted)

        ocr_col = []
        bbox_col = []
        exp_col = []

        for key in bbox_dict.keys():
            idx = int(key)
            ocr_col.append(ocr_dict_sorted[key])
            bbox_col.append(bbox_dict[key])
            exp_col.append(expert_name_dict[key])

        return {"ocr": ocr_col, "bbox": bbox_col, "experts": exp_col}

if __name__ == '__main__':
    ray.init(address='auto', runtime_env={"env_vars": {"PYTHONPATH": os.path.abspath('..'), "WORKING_DIRECTORY": os.path.abspath('..')}})
    client = serve.start()
    with open('../models/presence_clf.pkl', 'rb') as f:
            presence_clf = f.read()

    with open('../models/moe_clf.pkl', 'rb') as f:
            moe_clf = f.read()

    with open('../models/moe_labels.json', 'rb') as f:
            moe_labels = f.read()
    
    HeadBatchProcessor.deploy(presence_clf, moe_clf, moe_labels, expert_size='large')
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expert_size = 'large'
    parquet_fname = '../data/laion2b-en-10K.parquet'
    parquet_fname = '../data/laion2b-en-1K.parquet'
    parquet_result_fname = f'../data/laion2b-en-1K-experts-{expert_size}.parquet'
    dataset_path = '../data/laion2b-en-1K-large'

    dataset = load_dataset(dataset_path, parquet_fname, image_size=512, number_sample_per_shard=100)
    loader = DataLoader(dataset, batch_size=16, num_workers=4)

    future_results = []
    for batch in loader:
        batch = tuple(batch)
        future = client.get_handle("HeadBatchProcessor").process_batch.remote(batch)
        future_results.append(future)
        break

    results = ray.get(future_results[0])
    print(results)


