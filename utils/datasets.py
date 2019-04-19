# dataload
import os
import torch
import glob
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

    
class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        self.height = img_size
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        vid_formats = ['.mov', '.avi', '.mp4']

        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'File Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files
    
    
class MyLoadImages(Dataset):
    def __init__(self, path, img_size=416):
        self.height = img_size
        if os.path.isdir(path):
            self.files = os.listdir(path)
            self.path = path
        elif os.path.isfile(path):
            self.files = [path]
            self.path = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_file = self.files[idx]
        path = os.path.join(self.path, image_file) if self.path is not None else image_file
        img0 = cv2.imread(path)
        assert img0 is not None, 'File Not Found: {}'.format(path)
        
        img, _, _, _ = letterbox(img0, height=self.height) # img, (3, 416, 416)
        
        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return path, img, img0
        #plt.imshow(img.permute(1, 2, 0).numpy())