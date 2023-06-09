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
# Model architecture name
model_arch_name = "liif_edsr"
# Model arch config
in_channels = 3
encoder_channels = 64
out_channels = 3
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "LIIF_EDSR-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/DIV2K/LIIF/train"

    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    gt_image_size = int(upscale_factor * 48)
    batch_size = 16
    num_workers = 4

    # norm
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # The address to load the pretrained model
    pretrained_model_weights_path = ""

    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 1000
    eval_epoch = 1

    # Loss function weight
    loss_weights = 1

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [200, 400, 600, 800]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training results
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = "./results/Set5"
    gt_dir = "./data/Set5/GTmod12"

    # model_weights_path = "pretrain_model/LIIF_EDSR_x4-DIV2K-cc1955cd.pth.tar"
    model_weights_path = "pretrain_model/LIIF_EDSR_x4-DIV2K-cc1955cd.pth.tar"