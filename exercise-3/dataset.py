import glob
import random
import os

import cv2
import numpy as np
import torch
import torchvision.transforms

from torch.utils.data import Dataset
from torchvision.transforms import transforms

import utils
import config


class ImageDataset(Dataset):
    def __init__(self, gt_size, root, mode='train', upscale=False):
        self.gt_size = gt_size
        self.upscale = upscale
        # Transforms for low resolution images and high resolution images
        self.files = []
        self.trans_norm=transforms.Normalize(config.mean,config.std)
        self.root = root
        if mode == 'train':
            self.files = sorted(os.listdir(self.root))
        elif mode == 'val':
            self.files = sorted(os.listdir(self.root))
        elif mode == 'test':
            self.lq = sorted(os.listdir(os.path.join(self.root, "LRbicx4")))
            self.files = sorted(os.listdir(os.path.join(self.root, "GTmod12")))
        self.mode = mode

    def __getitem__(self, index):
        # img_hr = cv2.imread(self.files[index])
        # print(self.files[index])
        index = index%len(self.files)
        if self.mode == 'train':
            img_hr = cv2.imread(os.path.join(self.root, self.files[index])).astype(np.float32) / 255.
            gt_crop_image = utils.random_crop(img_hr, self.gt_size)
            gt_crop_image = utils.random_rotate(gt_crop_image, [90, 180, 270])
            gt_crop_image = utils.random_horizontally_flip(gt_crop_image, 0.5)
            img_hr = utils.random_vertically_flip(gt_crop_image, 0.5)
            img_lr = cv2.resize(img_hr, dsize=(self.gt_size // 4, self.gt_size // 4), interpolation=cv2.INTER_CUBIC)
        elif self.mode == 'val':
            img_hr = cv2.imread(os.path.join(self.root, self.files[index])).astype(np.float32) / 255.
            img_hr = utils.center_crop(img_hr, self.gt_size)
            img_lr = cv2.resize(img_hr, dsize=(self.gt_size // 4, self.gt_size // 4), interpolation=cv2.INTER_CUBIC)
        else:
            img_lr, img_hr = cv2.imread(os.path.join(self.root, "LRbicx4", self.lq[index])).astype(
                np.float32) / 255., cv2.imread(os.path.join(self.root, "GTmod12", self.files[index])).astype(
                np.float32) / 255.
            # print(img_hr.shape)
            # print(img_hr.shape)
        if self.upscale:
            img_lr = cv2.resize(img_lr, dsize=(img_hr.shape[1],img_hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        img_hr = utils.image_to_tensor(img_hr, False, False)
        img_lr = utils.image_to_tensor(img_lr, False, False)
        if self.mode != 'test':
            img_hr = self.trans_norm(img_hr)
            img_lr = self.trans_norm(img_lr)
        return {"lr": img_lr, "hr": img_hr, 'hr_path': self.files[index]}

    def __len__(self):
        if self.mode !='test':
            return len(self.files)*100
        else:
            return len(self.files)
