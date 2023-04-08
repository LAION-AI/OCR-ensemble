'''
This file is contributed by Ivan. To be re-factored

Installation:
conda create -n ocr python=3.10 numpy pillow transformers
conda activate ocr

#paddlepaddle quickstart guide is here https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/quickstart_en.md
#choose CPU/GPU versions here https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
#CPU only, I installed this, because some cudnn version problems: RuntimeError: (PreconditionNotMet) Cannot load cudnn shared library. Cannot invoke method cudnnGetVersion
pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
#GPU, cuda11.7
pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

pip install paddleocr

pip install salesforce-lavis

python single_file.py -i {your_image}
'''
import argparse
import time

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR
from lavis.models import load_model_and_preprocess

import numpy
import torch
from PIL import Image

# LAVIS blip and blip_vqa

blip_model, vis_processors = None, None


def caption_img(src_img, device=None):
    global blip_model, vis_processors
    # setup device to use
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if blip_model is None:
        blip_model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # load sample image
    if type(src_img) is str:
        raw_image = Image.open(src_img).convert("RGB")
    elif type(src_img) is numpy.ndarray:
        raw_image = Image.fromarray(src_img).convert('RGB')
    else:
        raw_image = src_img
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    return blip_model.generate({"image": image})


blip_vqa_model, vis_vqa_processors, txt_vqa_processors = None, None, None


def question_img(src_img, question='', device=None):
    global blip_vqa_model, vis_vqa_processors, txt_vqa_processors
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if blip_vqa_model is None:
        blip_vqa_model, vis_vqa_processors, txt_vqa_processors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    # load sample image
    if type(src_img) is str:
        raw_image = Image.open(src_img).convert("RGB")
    elif type(src_img) is numpy.ndarray:
        raw_image = Image.fromarray(src_img).convert('RGB')
    else:
        raw_image = src_img
    # ask a random question.
    # question = "Is it image contains text?"
    image = vis_vqa_processors["eval"](raw_image).unsqueeze(0).to(device)
    if type(question) is str:
        questions = [txt_vqa_processors["eval"](question)]
    else:
        questions = [txt_vqa_processors["eval"](q) for q in question]
    answers = [blip_vqa_model.predict_answers(
        samples={"image": image, "text_input": q}, inference_method="generate")[0] for q in questions]
    return answers[0] if type(question) is str else answers

# paddleocr


paddleocr_model = None


def ocr_img(src_img):
    # need to run only once to download and load model into memory
    global paddleocr_model
    if paddleocr_model is None:
        paddleocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    if type(src_img) is Image.Image:
        raw_img = numpy.asarray(src_img)
    else:
        raw_img = src_img
    # src_img is path or ndarray
    result = paddleocr_model.ocr(raw_img, cls=True)
    return result[0]

# trocr


trocr_processor = None
trocr_model = None


def ocr_handwritten(src_img, boxes=None, border=16):
    if type(src_img) is str:
        raw_img = Image.open(src_img).convert("RGB")
    elif type(src_img) is numpy.ndarray:
        raw_img = Image.fromarray(src_img).convert('RGB')
    else:
        raw_img = src_img
    is_parts = not boxes is None
    if boxes is None:
        w, h = raw_img.width, raw_img.height
        boxes = [[[0, 0], [w, 0], [w, h], [0, h]]]
    global trocr_processor, trocr_model
    if trocr_processor is None:
        trocr_processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-large-handwritten')
        trocr_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-large-handwritten')
    res = []
    for box in boxes:
        region = raw_img.crop(
            (box[0][0]-border, box[0][1]-border, box[2][0]+border, box[2][1]+border))
        pixel_values = trocr_processor(
            images=region, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        res.append([box, generated_text])
    return res if is_parts else res[0]
# combined


def ocr_with_vqa(src_img):
    if type(src_img) is str:
        raw_img = Image.open(src_img).convert("RGB")
    elif type(src_img) is numpy.ndarray:
        raw_img = Image.fromarray(src_img).convert('RGB')
    else:
        raw_img = src_img
    border = 8
    recognized = ocr_img(raw_img)
    boxes = [b[0] for b in recognized]
    for i, box in enumerate(boxes):
        region = raw_img.crop(
            (box[0][0]-border, box[0][1]-border, box[2][0]+border, box[2][1]+border))
        recognized[i].append(
            ['is_handwritten', question_img(region, 'Is it handwritten text?')])
        recognized[i][-1].append(ocr_handwritten(region)[1])
    return recognized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True,
                        type=str, help='Path to image')
    return parser.parse_args()


if __name__ == "__main__":
    i = parse_args().image
    start = time.time()
    caption = caption_img(i)
    results = ocr_with_vqa(i)
    end = time.time()
    print('Recognition results:')
    print(f'Caption:{caption}')
    print('OCR:')
    print(results)
    print(f'Time spent: {end-start}')