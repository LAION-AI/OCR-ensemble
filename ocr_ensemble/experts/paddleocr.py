from paddleocr import PaddleOCR
import torch
from matplotlib import pyplot as plt
import numpy as np

def paddle_transform(image):
    return image

class PaddleOCRExpert():
    def __init__(self, device="cpu", hparams={"use_angle_cls": False, "lang": "en", "ocr_version": "PP-OCRv3", "show_log":False}):
        if device:
            use_gpu = device=="cuda"
        else:
            use_gpu = torch.cuda.is_available()
        self.reader = PaddleOCR(use_gpu=use_gpu, **hparams)
    
    def get_transform(self):
        return paddle_transform

    def get_collate(self):
        return self.get_img_collate()

    def get_img_collate(self):
        def collate(imgs):
            return imgs
        return collate

    def process_batch(self, batch):
        return self.process_list(batch)

    def process_list(self, images):
        return [self(img) for img in images]

    def __call__(self, image):
        # make sure image is in the right format...
        #plt.imshow(image)
        #plt.show()
        xmax = image.shape[1]
        ymax = image.shape[0]
        img = image.copy()
        if img.max() == 1:
            img = 255*img
        res = self.reader.ocr(img, cls=False, det=False, rec=True)
        txts = [line[0] for line in res[0]]
        # detection + recognition
        #res = self.readler.ocr(img)
        #txts = [line[1][0] for line in res[0]]
        if len(txts) > 0:
            return ' '.join(txts)
        return ''
