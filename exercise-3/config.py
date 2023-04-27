# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/24 10:28
@Auth ： BHLL
@File ：config.py.py
@IDE ：PyCharm
@Motto:咕咕嘎嘎
"""

import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 2)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True

# Model arch config
in_channels = 3
encoder_channels = 64
out_channels = 3
upscale_factor = 4

train_root = 'data/DIV2k/train_hr'
val_root = 'data/DIV2k/valid_hr'
test_root = 'data/Set5'

# Optimizer parameter
model_lr = 1e-4
model_betas = (0.9, 0.99)
model_eps = 1e-8
model_weight_decay = 0.0
milestones = [10, 20, 40, 60]
gamma = 0.5
mean = [0.4488, 0.4371, 0.4040]
std = [1.0, 1.0, 1.0]
