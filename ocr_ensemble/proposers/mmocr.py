from mmocr.apis import TextDetInferencer
import torch, cv2
from utils import rotatedCrop
import numpy as np
import matplotlib.pyplot as plt

class MmocrProposer:
    def __init__(self, device=None, batch_size = 1, model='DBNetpp'):
        if device:
            use_gpu = device=="cuda"
        else:
            use_gpu = torch.cuda.is_available()
            
        self.reader = TextDetInferencer(model=model)
        self.batch_size = batch_size

    def __call__(self, image):
        # make sure image is in the right format...
        xmax = image.shape[1]
        ymax = image.shape[0]
        img = image.copy()
        if img.max() == 1:
            img = 255*img

        result = self.reader(img, batch_size=self.batch_size)
        for prediction in result['predictions']:
            bboxes = []
            polygons = prediction['polygons']
            scores = prediction['scores']
            for polygon, score in zip(polygons, scores):
                bbox = np.array(polygon).reshape(-1, 2)
                paddle_format = np.array([bbox[0], bbox[3], bbox[2], bbox[1]])
                bboxes.append({'bbox': paddle_format, 'score': score})

            bboxes.reverse()
            crops = [rotatedCrop(image, bbox['bbox']) for bbox in bboxes]            
            if len(crops) == 0:
                bboxes.append({'bbox': np.array([[0,0], [0, ymax], [xmax, ymax], [xmax, 0]]), 'score': 0})  
                crops += [image.copy()]
            return crops, bboxes
