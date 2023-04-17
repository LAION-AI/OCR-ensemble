from paddleocr import PaddleOCR
import torch
from matplotlib import pyplot as plt

class PaddleOCRExpert():
    def __init__(self, device=None, hparams={"use_angle_cls": False, "lang": "en", "ocr_version": "PP-OCRv3", "show_log":False}):
        if device:
            use_gpu = device=="cuda"
        else:
            use_gpu = torch.cuda.is_available()
        self.reader = PaddleOCR(use_gpu=use_gpu, **hparams)


    def process_list(self, images):
        return [self(img) for img in images]

    def __call__(self, image):
        # make sure image is in the right format...
        xmax = image.shape[1]
        ymax = image.shape[0]
        img = image.copy()
        if img.max() == 1:
            img = 255*img
        res = self.reader.ocr(img)
        txts = [line[1][0] for line in res[0]]
        if len(txts) > 0:
            return ' '.join(txts)
        return ''
