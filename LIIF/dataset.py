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
    def __init__(self, gt_size, root, mode='train'):
        self.gt_size = gt_size
        # Transforms for low resolution images and high resolution images
        self.files = []
        self.root = root
        if mode == 'train':
            self.files = sorted(os.listdir(self.root))
        elif mode == 'val':
            self.files = sorted(os.listdir(self.root))[:100]
        self.mode = mode

    def __getitem__(self, index):
        # img_hr = cv2.imread(self.files[index])
        # print(self.files[index])
        index = index % len(self.files)
        img_hr = cv2.imread(os.path.join(self.root, self.files[index])).astype(np.float32) / 255.
        if self.mode == 'train':
            gt_crop_image = utils.random_crop(img_hr, self.gt_size)
            gt_crop_image = utils.random_rotate(gt_crop_image, [90, 180, 270])
            gt_crop_image = utils.random_horizontally_flip(gt_crop_image, 0.5)
            img_hr = utils.random_vertically_flip(gt_crop_image, 0.5)
        elif self.mode == 'val':
            img_hr = utils.center_crop(img_hr, self.gt_size)
        img_lr = cv2.resize(img_hr, dsize=(self.gt_size // 4, self.gt_size // 4), interpolation=cv2.INTER_CUBIC)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        img_hr = utils.image_to_tensor(img_hr, False, False)
        img_lr = utils.image_to_tensor(img_lr, False, False)
        gt_tensor_coord = utils.make_coord(img_hr.contiguous().shape[-2:])
        gt_tensor_contiguous = img_hr.contiguous().view(3, -1).permute(1, 0)
        gt_tensor_cell = torch.ones_like(gt_tensor_coord)
        gt_tensor_cell[:, 0] *= 2 / img_hr.shape[-2]
        gt_tensor_cell[:, 1] *= 2 / img_hr.shape[-1]

        return {"gt": gt_tensor_contiguous, "cell": gt_tensor_cell, "coord": gt_tensor_coord, "lr": img_lr}

    def __len__(self):
        if self.mode == 'train':
            return len(self.files) * 20
        else:
            return len(self.files)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_gt_images_dir (str): ground truth image in test image
        test_lr_images_dir (str): low-resolution image in test image
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder

        self.gt_image_file_names = os.listdir(test_gt_images_dir)
        self.lr_image_file_names = os.listdir(test_lr_images_dir)
        self.lr_root = test_lr_images_dir
        self.gt_root = test_gt_images_dir

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        gt_image = cv2.imread(os.path.join(self.gt_root, self.gt_image_file_names[batch_index])).astype(
            np.float32) / 255.
        lr_image = cv2.imread(os.path.join(self.lr_root, self.lr_image_file_names[batch_index])).astype(
            np.float32) / 255.
        # print(lr_image.shape,gt_image.shape)
        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = utils.image_to_tensor(gt_image, False, False)
        lr_tensor = utils.image_to_tensor(lr_image, False, False)
        gt_tensor_coord = utils.make_coord(gt_tensor.contiguous().shape[-2:])
        gt_tensor_contiguous = gt_tensor.contiguous().view(3, -1).permute(1, 0)
        gt_tensor_cell = torch.ones_like(gt_tensor_coord)
        gt_tensor_cell[:, 0] *= 2 / gt_tensor.shape[-2]
        gt_tensor_cell[:, 1] *= 2 / gt_tensor.shape[-1]

        return {"gt": gt_tensor_contiguous, "cell": gt_tensor_cell, "coord": gt_tensor_coord, "lr": lr_tensor,
                'path': self.gt_image_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)
