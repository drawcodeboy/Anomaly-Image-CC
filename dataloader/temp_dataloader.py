from torch.utils.data import Dataset, DataLoader

import torch
from PIL import Image
import numpy as np
import cv2
import os

from .transform import TempTransforms

class TempDataset(Dataset):
    def __init__(self, data_dir,
                 transform=TempTransforms,
                 mode:str='train',
                 encoder:str='ResNet'):
        
        self.data_dir = data_dir
        self.data_li = [] # {image_path}
        
        self.transform = transform()
        self.mode = mode
        self.encoder = encoder
        
        self._check()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        mask = self.get_mask(self.data_li[idx])
        
        # mask To Tensor
        mask = mask.transpose(2, 0, 1)
        mask /= 255.0
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return self.transform(mask, self.mode), self.data_li[idx]
    
    def get_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        if self.encoder == 'ViT':
            mask = cv2.resize(mask, dsize=(224, 224))
        return mask
    
    def _check(self):
        file_cnt = 0
    
        for filename in os.listdir(self.data_dir):
            try:
                image = cv2.imread(os.path.join(self.data_dir, filename), cv2.IMREAD_UNCHANGED)
                self.data_li.append(os.path.join(self.data_dir, filename))
                file_cnt += 1
            except:
                print("Can\'t Open this file")
                pass
             
            print(f"\rLoad Data {file_cnt:04d} samples", end="")
        print()