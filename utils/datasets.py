# dataload
import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from .utils import xyxy2xywh, letterbox

class LoadDataset(Dataset):

    def __init__(self, path, img_size=416):
        with open(path, 'r') as f:
            self.images = f.read().splitlines()
            self.images = list(filter(lambda x: len(x) > 0, self.images))
        assert len(self.images) > 0, 'No images found in {}'.format(path)
        
        self.img_size = img_size
        self.labels = [x.replace('images', 'labels').replace('.bmp','.txt').replace('.jpg','.txt').replace('.png','txt') for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_pth = self.images[idx]
        label_pth = self.labels[idx]
        img = cv2.imread(img_pth) # BGR
        assert img is not None, 'File Not Found: {}'.format(img_pth)
        
        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.img_size)
        
        # load label
        labels = []
        with open(label_pth, 'r') as f:
            lines = f.read().splitlines()
        assert lines is not None, 'No annotations in: {}'.format(label_pth)
        
        x = np.array([x.split() for x in lines], dtype=np.float32)
        if x.size > 0:
            # shit xywh to pixel xyxy of padded.
            labels = x.copy()
            labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh
            
        nL = len(labels) # num of labels
        
        # convert xyxy to xywh
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size
            
        labels_out = torch.zeros((nL, 6))
        labels_out[:, 1:] = torch.from_numpy(labels)
        
        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_pth, (h, w)
        #plt.imshow(img.permute(1, 2, 0).numpy())
        
    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw