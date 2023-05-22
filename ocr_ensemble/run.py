import sys, os, cv2, torch, argparse
sys.path.append("..")

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
import uvicorn
from pydantic import BaseModel
from typing import List
from torch.utils.data import default_collate
import fastparquet as pq


class OCRItem(BaseModel):
    img: str
    bbox_idx: int
    key: str

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


app = FastAPI()
@app.on_event("startup")
async def startup_event():
    global presence, clip_emb, device, proposer, expert_text_dict, expert_dict, clf, batch_size, num_workers
    parser = argparse.ArgumentParser(description='OCR Ensemble')
    parser.add_argument('--device', type=str, default="cuda", help='Device to be used. Default: "cuda"')
    parser.add_argument('--expert_size', type=str, default='large', help='Size of expert. Default: "large"')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for DataLoader. Default: 200')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader. Default: 4')

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    expert_size, batch_size, num_workers = args.expert_size, args.batch_size, args.num_workers

    clip_emb = ClipEmbedding()
    presence = ClipPresence(clip_emb=clip_emb)
    proposer = PaddleOCRProposalGenerator(device='cpu')

    expert_text_dict = {f"trocr-{expert_size}-handwritten": "handwritten text, handwriting, black on white",
                "paddleocr": "text in a document, website, or presentation",
                f"trocr-{expert_size}-stage1": "text in a scene"}

    expert_dict = {f"trocr-{expert_size}-handwritten": HandwrittenExpert(expert_size),
               "paddleocr": PaddleOCRExpert(device='cpu'),
               f"trocr-{expert_size}-stage1": Stage1Expert(expert_size)}
    
    clf = ClipMulticlass(list(expert_text_dict.values()),
                     clip_emb = clip_emb)
    
    
@app.post("/process_batch")
async def process_batch_endpoint(batch: OCRBatch):
    results = process_batch(batch)
    return {"results": results}

def process_batch(batch):
    dataset_clip = deepcopy(batch)
    dataset_clip.map_tuple(presence.get_transform(), identity, identity)
    features, labels, label_dict  = [], [], {}

    for imgs, captions, keys in tqdm(dataset_clip):
        preds = presence.predict(imgs.to(device))
        labels += [preds]
        for key, pred in zip(keys, preds):
            label_dict[key] = pred

    labels = np.concatenate(labels, axis=0)

    batch_crops = deepcopy(batch)
    batch_crops.compose(partial(get_crops, label_dict=label_dict, proposer=proposer))
    batch_crops.map_tuple(clf.get_transform(), identity, identity)
    
    bbox_dict = defaultdict(list)
    bbox_label_dict = defaultdict(list)
    bbox_scores_dict = defaultdict(list)   

    for crops, bboxes, keys in tqdm(batch_crops):
        labels = clf.predict(crops.to(device))
        for label, bbox, key in zip(labels, bboxes, keys):
            bbox_dict[key] += [bbox]
            bbox_label_dict[key] += [label]
            #bbox_scores_dict[key] += [score.tolist()]

    ocr_dict = defaultdict(list)
    for idx, key in enumerate(expert_dict.keys()):
        print(f'Expert {key} ...')
        expert = expert_dict[key]
        batch_filtered = deepcopy(batch)
        batch_filtered.compose(partial(get_efficient_and_filtered_crops,
                                         expert_idx = idx, 
                                         bbox_dict = bbox_dict,
                                         bbox_label_dict = bbox_label_dict))
        batch_filtered.map_tuple(expert.get_transform(), identity, identity)
        if key is "paddleocr":
            collate_fn = partial(collate_crops, img_collate=expert.get_collate())
        else:
            collate_fn = collate

        for crops, bbox_ids, keys in tqdm(batch_filtered):
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

    ocr_col = ['']*len(batch)
    bbox_col = [[]]*len(batch)
    exp_col = ['']*len(batch)
    for key in bbox_dict.keys():
        idx = int(key)
        ocr_col[idx] = ocr_dict_sorted[key]
        bbox_col[idx] = bbox_dict[key]
        exp_col[idx] = expert_name_dict[key]

    return {"ocr": ocr_col, "bbox": bbox_col, "experts": exp_col}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)