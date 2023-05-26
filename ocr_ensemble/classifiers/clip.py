import pickle
import clip
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch 
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from functools import partial
import json

def clip_transform(img, preprocess):
    toPIL = T.ToPILImage()
    return T.Lambda(lambda img: preprocess(toPIL(rearrange(torch.tensor(img), 'a b c -> c a b'))))(img)

class ClipEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
    
    def get_transform(self):
        return partial(clip_transform, preprocess=self.preprocess)

    def get_transform_depr(self):
        def transform(img):
            assert img.shape[2] == 3 or img.dtype != np.uint8, "we expect uint8 HxWxC images with values in 0..255"
            return self.clip_preproc(img)
        return transform

    def get_collate(self):
        return self.get_img_collate()

    def get_img_collate(self):
        def collate(imgs):
            assert imgs[0].shape[2] == 3 or imgs[0].dtype != np.uint8, "we expect uint8 HxWxC images with values in 0..255"
            # how to turn a list of imgs to a batch
            batch = torch.cat([self.clip_preproc(img).unsqueeze(0) for img in imgs], axis=0)
            return batch
        return collate 

    @torch.no_grad()
    def encode_image_batch(self, batch, normalize=True): # note that batch should be on same device as model
        features = self.model.encode_image(batch.to(self.device))
        if normalize:
            features = F.normalize(features.float(), dim=1)
        return features

    @torch.no_grad()
    def encode_texts(self, texts, normalize=True):
        features = self.model.encode_text(clip.tokenize(texts).to(self.device))
        if normalize:
            features = F.normalize(features.float(), dim=1)
        return features


class ClipMulticlass():
    # the expets currently expect [height, width, channels], 0-255, uint8 HxWxC
    def __init__(self, expert_texts, clip_emb=None, debug=False, model_directory='../models/moe_clf_4.pkl', targets_directory='../models/moe_labels_4.json'):
        
        self.debug = debug
        self.expert_texts = [text[:77] for text in expert_texts] #:77 is the max length for openai ViT-B/32
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        if clip_emb:
            self.clip_emb = clip_emb 
        else:
            self.clip_emb = ClipEmbedding() 
        self.expert_embs = self.clip_emb.encode_texts(self.expert_texts)
        if self.debug:
            print(self.expert_texts)
            print(self.expert_embs)
        with open(model_directory, 'rb') as f:
            self.moe_clf = pickle.load(f) # sklearn logisticregression for now
        
        with open(targets_directory, 'r') as f:
            self.moe_targets = json.load(f)
            print('only supported targets', self.moe_targets)

    def get_transform(self):
        return self.clip_emb.get_transform()

    def get_collate(self):
        return self.get_img_collate()
    
    def get_img_collate(self):
        return self.clip_emb.get_img_collate()


    def predict(self, batch):
        batch = self.clip_emb.encode_image_batch(batch)
        return self.moe_clf.predict(batch.detach().cpu().numpy()) 

    def predict_proba(self, batch):
        batch = self.clip_emb.encode_image_batch(batch)
        return self.moe_clf.predict_proba(batch.detach().cpu().numpy()) 


class ClipMulticlassZeroshot():
    # the expets currently expect [height, width, channels], 0-255, uint8 HxWxC
    def __init__(self, expert_texts, clip_emb=None, debug=False):
        self.debug = debug
        self.expert_texts = [text[:77] for text in expert_texts] #:77 is the max length for openai ViT-B/32
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        if clip_emb:
            self.clip_emb = clip_emb 
        else:
            self.clip_emb = ClipEmbedding() 
        self.expert_embs = self.clip_emb.encode_texts(self.expert_texts)
        if self.debug:
            print(self.expert_texts)
            print(self.expert_embs)

    def get_transform(self):
        return self.clip_emb.get_transform()

    def get_collate(self):
        return self.get_img_collate()
    
    def get_img_collate(self):
        return self.clip_emb.get_img_collate()

    @torch.no_grad()
    def scores(self, batch): # note that batch should be on same device as model
        features = self.clip_emb.encode_image_batch(batch)
        scores = features @ self.expert_embs.T 
        return scores

    def predict(self, batch):
        return self.scores(batch).argmax(dim=1)

    def predict_proba(self, batch):
        return F.softmax(self.scores(batch), dim=1)


class ClipPresence():
    def __init__(self, clip_emb=None, model_directory='../models/presence_clf.pkl'):
        if clip_emb:
            self.clip_emb = clip_emb 
        else:
            self.clip_emb = ClipEmbedding()
        with open(model_directory, 'rb') as f:
            self.presence_clf = pickle.load(f) # sklearn logisticregression for now

    def get_transform(self):
        return self.clip_emb.get_transform()

    def get_collate(self):
        return self.get_img_collate()

    def get_img_collate(self):
        return self.clip_emb.get_img_collate()
    
    def predict(self, batch):
        batch = self.clip_emb.encode_image_batch(batch, normalize=False)
        return self.presence_clf.predict(batch.detach().cpu().numpy()) 
    
    def predict_proba(self, batch):
        batch = self.clip_emb.encode_image_batch(batch, normalize=False)
        return self.presence_clf.predict_proba(batch.detach().cpu().numpy())