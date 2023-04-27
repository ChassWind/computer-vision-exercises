# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/24 15:03
@Auth ： BHLL
@File ：train_SRCNN.py
@IDE ：PyCharm
@Motto:咕咕嘎嘎
"""
import torch
from torch.optim.lr_scheduler import MultiStepLR
# test on Set5: psnr: 27.75408, ssim: 0.79789
import wandb
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import calculate_psnr_pt, calculate_ssim_pt, tensor2img
from dataset import *
from model import SRCNN
import config


def train_srcnn_epoch(Net, optim, cri, train_loader):
    Net.train()
    LOSS = 0
    num = 0
    for i, data in enumerate(train_loader):
        lr = data['lr'].to(config.device)
        hr = data['hr'].to(config.device)
        hq = Net(lr)
        optim.zero_grad()
        loss = cri(hr, hq)
        LOSS = LOSS + loss.item() * lr.shape[0]
        num = num + lr.shape[0]
        loss.backward()
        optim.step()
    Net.eval()
    return LOSS / num


def test_srcnn(Net, loader, save_image=False):
    Net.eval()
    PSNR = 0
    SSIM = 0
    root = 'result/srcnn'
    with torch.no_grad():
        for i, data in enumerate(loader):
            lr = data['lr'].to(config.device)
            hr = data['hr'].to(config.device)
            hr_path = data['hr_path']

            hq = Net(lr)
            # print(lr.shape)
            # print(hr.shape)
            # print(hq.shape)
            psnr = calculate_psnr_pt(hr, hq, crop_border=4, test_y_channel=True)
            ssim = calculate_ssim_pt(hr, hq, crop_border=4, test_y_channel=True)
            PSNR += psnr.cpu().numpy()[0]
            SSIM += ssim.cpu().numpy()[0]
            if save_image:
                save_path = os.path.join(root, hr_path[0])
                cv2.imwrite(save_path, tensor2img(hq))
                wandb.log({
                    'input_lr': wandb.Image(tensor2img(lr, rgb2bgr=False)),
                    'SRGAN': wandb.Image(tensor2img(hq, rgb2bgr=False)),
                    'gt_hr': wandb.Image(tensor2img(hr, rgb2bgr=False))
                })
        PSNR = PSNR / len(loader)
        SSIM = SSIM / len(loader)
    Net.train()
    return PSNR, SSIM


def train_srcnn(epochs):
    Net = SRCNN(num_channels=3).to(config.device)
    optim = Adam(Net.parameters(), lr=config.model_lr, betas=config.model_betas, eps=config.model_eps,
                 weight_decay=config.model_weight_decay)
    trainLoader = DataLoader(dataset=ImageDataset(mode='train', gt_size=256, root=config.train_root, upscale=True),
                             batch_size=32)
    valLoader = DataLoader(dataset=ImageDataset(mode='val', gt_size=256, root=config.val_root, upscale=True),
                           batch_size=1)
    testLoader = DataLoader(dataset=ImageDataset(mode='test', gt_size=-1, root=config.test_root, upscale=True),
                            batch_size=1)
    cri = nn.MSELoss()
    best_psnr, best_ssim = 0, 0
    lr_scheduler = MultiStepLR(optim, milestones=config.milestones, gamma=config.gamma)
    for epoch in range(epochs):

        epoch_loss = train_srcnn_epoch(Net, optim, cri, trainLoader)
        lr_scheduler.step()
        psnr, ssim = test_srcnn(Net, valLoader)
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
            print("save ok")
            torch.save(Net.state_dict(), "pretrain_model/srcnn.pth")
        wandb.log({'lr': optim.param_groups[0]['lr'],
                   "epoch": epoch,
                   "loss": epoch_loss,
                   'val_psnr': psnr,
                   'best_psnr': best_psnr,
                   'val_ssim': ssim,
                   'best_ssim': best_ssim
                   })
    Net.load_state_dict(torch.load("pretrain_model/srcnn.pth"))
    test_psnr, test_ssim = test_srcnn(Net, testLoader, save_image=True)
    print(test_psnr, test_ssim)
    print(type(test_psnr), type(test_ssim))
    print("test on Set5: psnr: {:.5f}, ssim: {:.5f}".format(test_psnr, test_ssim))


if __name__ == '__main__':
    wandb.init(project='SRCNN', )
    train_srcnn(80)
