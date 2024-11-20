import torch
import torchvision
import cv2
import numpy as np

import random
import copy

__all__ = ["TempTransforms"]
        
class RandomPixel:        
    def __call__(self, sample):
        height, width, channels = 64, 64, 3

        # 랜덤으로 한 픽셀 선택
        random_channel = random.randint(0, channels - 1)
        random_x = random.randint(0, width - 1)
        random_y = random.randint(0, height - 1)

        # 기존 이미지에 점 찍기 (예: 1로 덮어쓰기)
        sample[random_channel, random_y, random_x] = 1.0
        return sample
        
class TempTransforms:
    def __init__(self):
        
        self.train_transform = [
            # torchvision.transforms.RandomRotation(degrees=(-90, 90)),
            torchvision.transforms.ColorJitter(brightness=.5, hue=.3),
            RandomPixel()
        ]
        
        self.test_transform = [
        ]
        
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)
        
    def __call__(self, x, mode):
        if mode == 'train':
            return self.train_transform(x), self.train_transform(x)
        elif mode == 'test':
            return self.test_transform(x)