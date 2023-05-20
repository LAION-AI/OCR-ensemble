import sys, os, cv2, torch, argparse
sys.path.append("..")
from ocr_ensemble.data import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict
from copy import deepcopy
from paddleocr import draw_ocr
from PIL import Image

from ocr_ensemble.classifiers import ClipEmbedding, ClipMulticlass, ClipPresence
from ocr_ensemble.proposers import PaddleOCRProposalGenerator
from ocr_ensemble.proposers import rotatedCrop
from ocr_ensemble.experts import HandwrittenExpert, Stage1Expert, PaddleOCRExpert
from ocr_ensemble.data import identity

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
from torch.utils.data import default_collate
import fastparquet as pq
from ocr_ensemble.data import load_dataset


class OCRItem(BaseModel):
    img: str
    bbox_idx: int
    label: str

class OCRBatch(BaseModel):
    batch: List[OCRItem]

def collate(batch):
    imgs = []
    bboxes = []
    labels = []
    for img, bbox, label in batch:
        imgs += [img]
        bboxes += [bbox]
        labels += [label]
    return default_collate(imgs), bboxes, labels

def get_crops(src, label_dict, proposer):
    for img, caption, key in tqdm(src):
        if label_dict[key] == 1:
            crops, bboxes = proposer(img)
            for crop, bbox in zip(crops, bboxes):
                yield crop, bbox, key

def get_imgs_containing_text(src, label_dict):
    for img, caption, key in tqdm(src):
        if label_dict[key] == 1:
            yield img, caption, key

def get_efficient_and_filtered_crops(src, expert_idx, bbox_dict, bbox_label_dict):
    for img, caption, key in tqdm(src):
        bboxes = bbox_dict[key]
        labels = bbox_label_dict[key]
        for bbox_idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            if label == expert_idx:
                crop = rotatedCrop(img, bbox)
                yield crop, bbox_idx, key

def collate_crops(batch, img_collate):
    crops = []
    bboxes = []
    keys = []
    for crop, bbox, key in batch:
        crops += [crop]
        bboxes += [bbox]
        keys += [key]
    return img_collate(crops), bboxes, keys

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expert_size = "large"  # Or load from an environment variable or a config file
    model = Stage1Expert(expert_size)
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    app = FastAPI()
    @app.on_event("startup")
    async def startup_event():
        global model
        model = load_model()

    @app.post("/process_batch")
    async def process_batch(batch: OCRBatch):
        results = []
        for item in batch.batch:
            result = model.process_batch(item.img)
            results.append({"bbox_idx": item.bbox_idx, "key": item.key, "result": result})

        return {"results": results}
    
    
    parser = argparse.ArgumentParser(description='OCR Ensemble')
    parser.add_argument('--device', type=str, default="cuda", help='Device to be used. Default: "cuda"')
    parser.add_argument('--expert_size', type=str, default='large', help='Size of expert. Default: "large"')
    parser.add_argument('--dataset_path', type=str, default='./data/laion2b-en-1K-large', help='Path to the dataset. Default: "../data/laion2b-en-1K-large"')
    parser.add_argument('--parquet_fname', type=str, default='./data/laion2b-en-10K.parquet', help='Path to the parquet file. Default: "../data/laion2b-en-10K.parquet"')
    parser.add_argument('--parquet_result_fname', type=str, default='./data/laion2b-en-1K-experts-large.parquet', help='Path to the result parquet file. Default: "../data/laion2b-en-1K-experts-large.parquet"')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for the dataset. Default: 512')
    parser.add_argument('--number_sample_per_shard', type=int, default=100, help='Number of samples per shard for the dataset. Default: 100')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for DataLoader. Default: 200')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader. Default: 4')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    expert_size, parquet_fname, parquet_result_fname, dataset_path, batch_size, num_workers, image_size, num_samples_PS = args.expert_size, args.parquet_fname, args.parquet_result_fname,\
                                                                    args.dataset_path, args.batch_size, args.num_workers, args.image_size, args.number_sample_per_shard
    
    
    dataset = load_dataset(dataset_path, parquet_fname, image_size=image_size, number_sample_per_shard=num_samples_PS)

    clip_emb = ClipEmbedding()
    presence = ClipPresence(clip_emb=clip_emb)

    dataset_clip = deepcopy(dataset)
    dataset_clip.map_tuple(presence.get_transform(), identity, identity)

    loader = DataLoader(dataset_clip, batch_size=batch_size, num_workers=num_workers)

    features, labels, label_dict  = [], [], {}

    for imgs, captions, keys in tqdm(loader):
        preds = presence.predict(imgs.to(device))
        labels += [preds]
        for key, pred in zip(keys, preds):
            label_dict[key] = pred

    labels = np.concatenate(labels, axis=0)

    proposer = PaddleOCRProposalGenerator(device='cpu')
    dataset_crops = deepcopy(dataset)
    dataset_crops.compose(partial(get_crops, label_dict=label_dict, proposer=proposer))

    expert_text_dict = {f"trocr-{expert_size}-handwritten": "handwritten text, handwriting, black on white",
                    "paddleocr": "text in a document, website, or presentation",
                    f"trocr-{expert_size}-stage1": "text in a scene"}
    
    expert_dict = {f"trocr-{expert_size}-handwritten": HandwrittenExpert(expert_size),
                   "paddleocr": PaddleOCRExpert(device='cpu'),
                   f"trocr-{expert_size}-stage1": Stage1Expert(expert_size)}

    clf = ClipMulticlass(list(expert_text_dict.values()),
                         clip_emb = clip_emb)

    dataset_crops.map_tuple(clf.get_transform(), identity, identity)
    loader_crops = DataLoader(dataset_crops, 
                          batch_size=batch_size,
                          collate_fn=collate) # here num_worker breaks things in non-trivial ways
    
    bbox_dict = defaultdict(list)
    bbox_label_dict = defaultdict(list)
    bbox_scores_dict = defaultdict(list)

    for crops, bboxes, keys in tqdm(loader_crops):
        labels = clf.predict(crops.to(device))

        for label, bbox, key in zip(labels, bboxes, keys):
            bbox_dict[key] += [bbox]
            bbox_label_dict[key] += [label]
            #bbox_scores_dict[key] += [score.tolist()]


    ocr_dict = defaultdict(list)
    for idx, key in enumerate(expert_dict.keys()):
        print(f'Expert {key} ...')
        expert = expert_dict[key]
        dataset_filtered = deepcopy(dataset)
        dataset_filtered.compose(partial(get_efficient_and_filtered_crops,
                                         expert_idx = idx, 
                                         bbox_dict = bbox_dict,
                                         bbox_label_dict = bbox_label_dict))
        dataset_filtered.map_tuple(expert.get_transform(), identity, identity)
        if key is "paddleocr":
            collate_fn = partial(collate_crops, img_collate=expert.get_collate())
        else:
            collate_fn = collate
        loader_expert = DataLoader(dataset_filtered, 
                               batch_size=batch_size,
                               collate_fn=collate_fn)


        for crops, bbox_ids, keys in tqdm(loader_expert):
            texts = expert.process_batch(crops)
            for text, bbox_idx, key in zip(texts, bbox_ids, keys):
                ocr_dict[key] += [(bbox_idx, text)]

    ocr_dict_sorted = defaultdict(list)
    for key in ocr_dict.keys():
        list_of_tuples = ocr_dict[key]
        sorted_list_of_tuples = sorted(list_of_tuples, key=lambda x: x[0])
        ocr_dict_sorted[key] = [t[1] for t in sorted_list_of_tuples]

    expert_name_dict = defaultdict(list)
    expert_names = list(expert_dict.keys())
    for key, val in bbox_label_dict.items():
        expert_name_dict[key] = [expert_names[idx] for idx in val]

    df = pd.read_parquet(parquet_fname, engine='pyarrow')

    ocr_col = ['']*len(df)
    bbox_col = [[]]*len(df)
    exp_col = ['']*len(df)

    for key in bbox_dict.keys():
        idx = int(key)
        ocr_col[idx] = ocr_dict_sorted[key]
        bbox_col[idx] = bbox_dict[key]
        exp_col[idx] = expert_name_dict[key]

    df['OCR_BBOXES'] = bbox_col
    df['OCR_EXPERTS'] = exp_col
    df['OCR_TEXT'] = ocr_col

    df['OCR_BBOXES'] = df['OCR_BBOXES'].astype(str)
    df['OCR_EXPERTS'] = df['OCR_EXPERTS'].astype(str)
    df['OCR_TEXT'] = df['OCR_TEXT'].astype(str)
    pq.write(os.path.abspath(parquet_result_fname), df)

    handwritten_idcs = []
    for idx, exps in enumerate(exp_col):
        if f'trocr-{expert_size}-handwritten' in exps:
            handwritten_idcs += [idx]
            cv2.imshow('Image', Image(url=df.iloc[idx]['URL']))
            print(df.iloc[idx]['OCR_TEXT'])
            print(df.iloc[idx]['OCR_EXPERTS'])
            print()
            
    print(handwritten_idcs)
