import os

import cv2
import torch
from model import LIIF
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import wandb
import config
import utils
from dataset import ImageDataset, TestImageDataset
from utils import calculate_psnr_pt, calculate_ssim_pt


def train_epoch(train_loader, model, optimizer):
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(config.device)
    inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(config.device)

    gt_sub = torch.FloatTensor([0.5]).view(1, 1, -1).to(config.device)
    gt_div = torch.FloatTensor([0.5]).view(1, 1, -1).to(config.device)

    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(config.device)

        inp = (batch['lr'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.item()


def train():
    train_root = 'data/DIV2k/train_hr'
    val_root = 'data/DIV2k/valid_hr'
    train_loader = DataLoader(dataset=ImageDataset(config.gt_image_size, train_root, mode='train'),
                              batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(dataset=ImageDataset(config.gt_image_size, val_root, mode='val'),
                            batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = LIIF(in_channels=config.in_channels,
                 encoder_channels=config.encoder_channels,
                 out_channels=config.out_channels,
                 encoder_arch="edsr").to(config.device)
    optimizer = optim.Adam(model.parameters(),
                           config.model_lr,
                           config.model_betas,
                           config.model_eps,
                           config.model_weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=config.lr_scheduler_milestones, gamma=config.lr_scheduler_gamma)
    best_psnr, best_ssim = 0, 0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(train_loader, model, optimizer)
        lr_scheduler.step()
        if epoch % config.eval_epoch == 0:
            psnr, ssim = test_liif(val_loader, model)
            if psnr > best_psnr:
                best_psnr = psnr
                best_ssim = ssim
            wandb.log({'epoch': epoch,
                       "val_psnr": psnr,
                       'val_ssim': ssim,
                       })
        if epoch % 100 == 0:
            torch.save(model.state_dict(), "pretrain_model/liif_" + str(epoch) + ".pth")
        wandb.log({"epoch": epoch,
                   'lr': optimizer.param_groups[0]['lr'],
                   "loss": train_loss,
                   'best_psnr': best_psnr,
                   'best_ssim': best_ssim
                   })
    print('train end')


def test():
    model = LIIF(in_channels=config.in_channels,
                 encoder_channels=config.encoder_channels,
                 out_channels=config.out_channels,
                 encoder_arch="edsr").to(config.device)
    model.load_state_dict(torch.load(config.model_weights_path))
    test_loader = DataLoader(dataset=TestImageDataset(config.gt_dir, config.lr_dir),
                             batch_size=1, shuffle=False, num_workers=1)
    test_psnr, test_ssim = test_liif(test_loader, model, save_img=True)
    print(test_psnr, test_ssim)
    print("test on Set5: psnr: {:.5f}, ssim: {:.5f}".format(test_psnr, test_ssim))


def test_liif(dataloader, model, save_img=False):
    model.eval()
    # print(save_img)
    PSNR = utils.Averager()
    SSIM = utils.Averager()
    inp_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(config.device)
    inp_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to(config.device)

    gt_sub = torch.FloatTensor([0.5]).view(1, 1, -1).to(config.device)
    gt_div = torch.FloatTensor([0.5]).view(1, 1, -1).to(config.device)
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if k != 'path':
                    batch[k] = v.to(config.device)
            inp = (batch['lr']-inp_sub)/inp_div
            pred = model(inp, batch['coord'], batch['cell'])
            gt = batch['gt']
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)
            batch_size, channels, lr_image_height, lr_image_width = inp.shape

            shape = [batch_size,
                     round(lr_image_height * config.upscale_factor),
                     round(lr_image_width * config.upscale_factor),
                     channels]

            pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
            gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()
            psnr, ssim = calculate_psnr_pt(pred, gt, crop_border=4,
                                           test_y_channel=True).cpu().numpy(), calculate_ssim_pt(pred, gt,
                                                                                                 crop_border=4,
                                                                                                 test_y_channel=True).cpu().numpy()
            # print(psnr[0],ssim[0])
            if save_img:
                sr_image_path = os.path.join(config.sr_dir, batch['path'][0])
                sr_image = utils.tensor_to_image(pred, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sr_image_path, sr_image)
            PSNR.add(psnr[0], gt.shape[0])
            SSIM.add(ssim[0], gt.shape[0])
    return PSNR.item(), SSIM.item()


if __name__ == '__main__':
    wandb.init(project='LIIF', name='train_on_div2k_remake')
    train()

    # test()
