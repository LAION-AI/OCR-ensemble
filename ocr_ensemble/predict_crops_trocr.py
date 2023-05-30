import os
import argparse
import numpy as np
from PIL import Image
from .experts import HandwrittenExpert, PrintedExpert, SceneExpert, Stage1Expert
import pickle
import json
from tqdm import tqdm

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
    parser.add_argument('--type', type=str, default='stage1', choices=['handwritten', 'printed', 'scene', 'stage1'], help='Type of TrOCRExpert to use')
    parser.add_argument('--size', type=str, default='small', choices=['base', 'large'], help='Size to use for the TrOCRExpert')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size to use for the TrOCRExpert')
    parser.add_argument('--output', type=str, default='output.json', help='Path to the output file')
    args = parser.parse_args()

    # Create the appropriate TrOCRExpert
    if args.type == 'handwritten':
        trocr_expert = HandwrittenExpert(args.size)
    elif args.type == 'printed':
        trocr_expert = PrintedExpert(args.size)
    elif args.type == 'scene':
        trocr_expert = SceneExpert(args.size)
    elif args.type == 'stage1':
        trocr_expert = Stage1Expert(args.size)
    else:
        raise ValueError('Invalid type. Must be one of [handwritten, printed, scene, stage1]')

    # load images from the specified folder
    if args.input.endswith('.pkl'):
        with open(args.input, 'rb') as f:
            images = pickle.load(f)
    else:
        images = load_images_from_folder(args.input)
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    images = [np.array(img.convert('RGB')) for img in images]

    # run TrOCRExpert over all the loaded images
    batch = []
    res = []
    for image in tqdm(images):
        batch += [image]
        if len(batch) % args.batch_size == 0:
            texts = trocr_expert.process_list(batch)
            batch = []
            res += texts
            print(texts)
    if len(batch) > 0:
        texts = trocr_expert.process_list(batch)
        batch = []
        res += texts
    
    with open(args.output, 'w') as f:
        json.dump(res, f)

if __name__ == "__main__":
    main()
