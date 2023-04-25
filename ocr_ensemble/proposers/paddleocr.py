from paddleocr import PaddleOCR
from .utils import applyPadding, uull2xywh, rotatedCrop
import torch

class PaddleOCRProposalGenerator():
    def __init__(self, device=None, hparams={"use_angle_cls": False, "lang": "en", "ocr_version": "PP-OCRv3", "show_log":False}):
        # on the use_angle_cls parameter: https://stackoverflow.com/questions/74341531/use-angle-cls-and-cls-arguments-in-paddleocr
        if device:
            use_gpu = device=="cuda"
        else:
            use_gpu = torch.cuda.is_available()
        self.reader = PaddleOCR(use_gpu=use_gpu, **hparams)

    def __call__(self, image):
        # make sure image is in the right format...
        xmax = image.shape[1]
        ymax = image.shape[0]
        img = image.copy()
        if img.max() == 1:
            img = 255*img
        result = self.reader.ocr(img, cls=False, det=True, rec=False)
        # compute unrotated crops 
        bboxes = []
        for row in result:
            for box in row:
                bboxes += [box]
        bboxes.reverse()

        crops = [rotatedCrop(image, bbox) for bbox in bboxes]            
        # bboxes are in paddleocrs: top left, top right, bottom right, bottom left coordinate format
        if len(crops) == 0:
            #bboxes += [[[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]]]
            bboxes += [[[0,0], [0, ymax], [xmax, ymax], [xmax, 0]]]
            crops += [image.copy()]
        return crops, bboxes
