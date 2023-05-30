import os
import argparse
import numpy as np
from PIL import Image
from .experts import HandwrittenExpert, PrintedExpert, SceneExpert, Stage1Expert
from .classifiers import ClipMulticlass, ClipEmbedding
import pickle
import json
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import copy

# crop pickle file: https://drive.google.com/file/d/1t8UC0MmMcEL3z4VrXqjh66j-QUXmK09x/view?usp=sharing
# run using python -m ocr_ensemble.predict_crops .\data\label_studio\both_crops_bgr.pkl

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Add or remove file extension based on your need
            img = Image.open(os.path.join(folder, filename))
            img = np.array(img)
            if img is not None:
                images.append(img)
    return images

def main():
    parser = argparse.ArgumentParser(description='Run TrOCRExpert over a folder of image crops')
    parser.add_argument('input', type=str, help='Path to the folder containing images or pkl file')
    parser.add_argument('--size', type=str, default='large', choices=['base', 'large'], help='Size to use for the TrOCRExpert')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use for the TrOCRExpert')
    parser.add_argument('--output', type=str, default='output.json', help='Path to the output file')
    parser.add_argument('--show_hist', action='store_true', help='Show histogram of expert predictions')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    args = parser.parse_args()


    expert_size = args.size
    expert_text_dict = {f"trocr-{expert_size}-handwritten": "handwritten text, handwriting, black on white",
                        f"trocr-{expert_size}-printed": "text in a document, website, or presentation",
                        f"trocr-{expert_size}-str": "text in a scene",
                        f"trocr-{expert_size}-stage1": "rendered text"}
    expert_dict = {f"trocr-{expert_size}-handwritten": HandwrittenExpert,
                f"trocr-{expert_size}-printed": PrintedExpert,
                f"trocr-{expert_size}-str": SceneExpert,
                f"trocr-{expert_size}-stage1": Stage1Expert}
    expert_data = {f"trocr-{expert_size}-handwritten": [],
                        f"trocr-{expert_size}-printed": [],
                        f"trocr-{expert_size}-str": [],
                        f"trocr-{expert_size}-stage1": []}
    expert_preds = {f"trocr-{expert_size}-handwritten": [],
                        f"trocr-{expert_size}-printed": [],
                        f"trocr-{expert_size}-str": [],
                        f"trocr-{expert_size}-stage1": []}
    expert_keys = list(expert_dict.keys())
    
    clip_emb = ClipEmbedding()
    clf = ClipMulticlass(list(expert_text_dict.values()),
                        clip_emb = clip_emb,
                        model_directory='models/moe_clf_4.pkl', targets_directory='models/moe_labels_4.json')
    
    # load images from the specified folder
    if args.input.endswith('.pkl'):
        with open(args.input, 'rb') as f:
            images = pickle.load(f)
    else:
        images = load_images_from_folder(args.input)
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    images = [img.convert('RGB') for img in images] # clip wants PIL images

    if args.debug:
        images = images[:20]
    
    # run classifier
    clf_collate = clf.get_collate()
    batch = []
    res = []
    for image in tqdm(images):
        batch += [image]
        if len(batch) % args.batch_size == 0:
            expert_ids = clf.predict(clf_collate(copy.deepcopy(batch)))
            batch = []
            res += expert_ids.tolist()
    if len(batch) > 0:
        expert_ids = clf.predict(clf_collate(copy.deepcopy(batch)))
        batch = []
        res += expert_ids.tolist()
    expert_ids = np.array(res)
    del clf

    if args.show_hist:
        plt.hist(expert_ids, bins='rice')
        plt.show()

    # forward images to experts
    for expert_id, image in zip(expert_ids, images):
        expert_key = expert_keys[expert_id]
        expert_data[expert_key] += [np.array(image)]
    
    # run experts   
    for key, data in expert_data.items():
        batch = []
        expert = expert_dict[key](expert_size)
        for image in tqdm(data):
            batch += [image]
            if len(batch) % args.batch_size == 0:
                texts = expert.process_list(batch)
                batch = []
                expert_preds[key] += texts
        if len(batch) > 0:
            texts = expert.process_list(batch)
            batch = []
            expert_preds[key] += texts
        del expert

    # combine results
    result = {'preds':[], 'expert_keys':[]}
    for expert_id in expert_ids:
        expert_key = expert_keys[expert_id]
        result['preds'] += [expert_preds[expert_key].pop(0)]
        result['expert_keys'] += [expert_key]
    
    with open(args.output, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
