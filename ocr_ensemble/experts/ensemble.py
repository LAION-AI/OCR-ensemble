import pickle
import clip
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch 
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt

class Ensemble():
    # the expets currently expect [height, width, channels], 0-255, uint8
    def __init__(self, expert_dict = {}, debug=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.toPIL = T.ToPILImage()
        self.clip_preproc = T.Lambda(lambda img: self.preprocess(self.toPIL(rearrange(torch.tensor(img), 'a b c -> c a b'))))
        self.expert_dict = expert_dict
        self.expert_keys = list(expert_dict.keys()) 
        self.expert_embs = self.model.encode_text(clip.tokenize(self.expert_keys).to(self.device)).cpu().detach()
        self.expert_embs = F.normalize(self.expert_embs.float(), dim=1)
        self.debug = debug
        if self.debug:
            print(self.expert_embs)

    
    def process_list(self, crop_list): 
        # crop list is a list of image crops that we think contain text
        result = len(crop_list)*['']
        # batch
        dataset = TensorDataset(torch.cat([self.clip_preproc(img).unsqueeze(0) for img in crop_list], axis=0))
        loader = DataLoader(dataset, batch_size=50)
        # compute clip embeddings
        features = []
        for batch in loader:
            feat = self.model.encode_image(batch[0].to(self.device))
            features += [feat.cpu().detach()]
        features = torch.cat(features, axis=0)
        features = F.normalize(features.float(), dim=1)
        # compute expert ranking
        scores = np.dot(features.numpy(), self.expert_embs.numpy().T)
        expert_ids = np.argmax(scores, axis=1)
        # forward to expert       
        for idx in np.unique(expert_ids):
            mask = expert_ids == idx 
            mask_idcs = np.where(mask)[0]
            img_list = [crop_list[img_idx] for img_idx in mask_idcs]
            ocrs = []
            if self.debug:
                print(self.expert_keys[idx], ":")
                for img in img_list:
                    plt.imshow(img)
                    plt.show()
                    ocrs += [self.expert_dict[self.expert_keys[idx]](img)]
                    print(ocrs[-1])
            else:
                ocrs = self.expert_dict[self.expert_keys[idx]].process_list(img_list)
            for img_idx, ocr in zip(mask_idcs, ocrs):
                result[img_idx] += 'expert %d: %s'%(idx, ocr)
        return result

    def __call__(self, image):
        return self.process_list([image])[0]
            
