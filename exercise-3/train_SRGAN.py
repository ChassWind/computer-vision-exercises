# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/23 22:46
@Auth ： BHLL
@File ：train_SRGAN.py
@IDE ：PyCharm
@Motto:咕咕嘎嘎
"""
from torch.optim.lr_scheduler import MultiStepLR

from utils import calculate_psnr_pt, calculate_ssim_pt, tensor2img
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *
from dataset import *
import wandb


# test on Set5: psnr: 26.52058, ssim: 0.79820
# test on Set5: psnr: 22.12204, ssim: 0.78106

def train_srgan_epoch(generator, discriminator, feature_extractor, optimizer_G, optimizer_D, criterion_content,
                      criterion_pixel, criterion_GAN, trainLoader):
    generator.train()
    discriminator.train()
    lossG = 0
    lossD = 0
    num = 0
    for i, imgs in enumerate(trainLoader):
        # Configure model input
        imgs_lr = imgs['lr'].to(config.device)
        imgs_hr = imgs['hr'].to(config.device)

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))).to(config.device),
                         requires_grad=False)
        fake = Variable(torch.FloatTensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))).to(config.device),
                        requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        discriminator.eval()
        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # pixel loss
        loss_pix = criterion_pixel(imgs_hr, gen_hr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = 1e-2 * loss_pix + loss_content + 1e-3 * loss_GAN
        lossG = lossG + loss_G.item() * imgs_lr.shape[0]
        num += imgs_lr.shape[0]

        loss_G.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        discriminator.train()
        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        lossD = lossD + loss_D.item() * imgs_lr.shape[0]
        loss_D.backward()
        optimizer_D.step()
    generator.eval()
    return lossG / num, lossD / num


def test_srgan(generator, loader, save_image=False):
    generator.eval()
    PSNR = 0
    SSIM = 0
    root = 'result/srgan'
    with torch.no_grad():
        for i, data in enumerate(loader):
            lr = data['lr'].to(config.device)
            hr = data['hr'].to(config.device)
            hr_path = data['hr_path']
            hq = generator(lr)
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
    generator.train()
    return PSNR, SSIM


def train_srgan(epochs):
    device = config.device
    generator = GeneratorResNet().to(device)
    discriminator = Discriminator(input_shape=(3, 256, 256)).to(device)
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    generator.load_state_dict(torch.load("pretrain_model/generator.pth"))
    # Losses
    criterion_GAN = torch.nn.MSELoss().to(device=device)
    criterion_content = torch.nn.L1Loss().to(device=device)
    criterion_pixel = torch.nn.L1Loss().to(device=device)

    trainLoader = DataLoader(ImageDataset(gt_size=256, mode='train', root=config.train_root, upscale=False),
                             batch_size=16, shuffle=True,
                             num_workers=4)
    valLoader = DataLoader(ImageDataset(gt_size=256, mode='val', root=config.val_root, upscale=False), batch_size=1,
                           num_workers=1,
                           shuffle=False)
    testLoader = DataLoader(ImageDataset(gt_size=-1, mode='test', root=config.test_root, upscale=False), batch_size=1,
                            shuffle=False,
                            num_workers=1)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.model_lr, betas=config.model_betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.model_lr, betas=config.model_betas)
    sch1 = MultiStepLR(optimizer_G, milestones=config.milestones, gamma=config.gamma)
    sch2 = MultiStepLR(optimizer_D, milestones=config.milestones, gamma=config.gamma)
    best_psnr, best_ssim = 0, 0
    for epoch in range(epochs):
        lossG, lossD = train_srgan_epoch(generator, discriminator, feature_extractor, optimizer_G, optimizer_D,
                                         criterion_content, criterion_GAN, criterion_pixel, trainLoader)
        psnr, ssim = test_srgan(generator, valLoader)
        sch1.step()
        sch2.step()
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
            torch.save(generator.state_dict(), "pretrain_model/srgan_generator.pth")

        wandb.log({"epoch": epoch,
                   "G_lr": optimizer_G.param_groups[0]['lr'],
                   "D_lr": optimizer_D.param_groups[0]['lr'],
                   "lossG": lossG,
                   'lossD': lossD,
                   'val_psnr': psnr,
                   'best_psnr': best_psnr,
                   'val_ssim': ssim,
                   'best_ssim': best_ssim
                   })

    generator.load_state_dict(torch.load("pretrain_model/srgan_generator.pth"))
    test_psnr, test_ssim = test_srgan(generator, testLoader, save_image=True)
    print("test on Set5: psnr: {:.5f}, ssim: {:.5f}".format(test_psnr, test_ssim))


def train_srresnet_epoch(generator, optim, cri, train_loader):
    generator.train()
    LOSS = 0
    num = 0
    for i, data in enumerate(train_loader):
        lr = data['lr'].to(config.device)
        hr = data['hr'].to(config.device)
        hq = generator(lr)
        optim.zero_grad()
        loss = cri(hr, hq)
        LOSS = LOSS + loss.item() * lr.shape[0]
        num = num + lr.shape[0]
        loss.backward()
        optim.step()
    generator.eval()
    return LOSS / num


def test_srresnet(Net, loader, save_image=False):
    Net.eval()
    PSNR = 0
    SSIM = 0
    root = 'result/srresnet'
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
                    'srresnet_input_lr': wandb.Image(tensor2img(lr, rgb2bgr=False)),
                    'srresnet_SRGAN': wandb.Image(tensor2img(hq, rgb2bgr=False)),
                    'srresnet_gt_hr': wandb.Image(tensor2img(hr, rgb2bgr=False))
                })
        PSNR = PSNR / len(loader)
        SSIM = SSIM / len(loader)
    Net.train()
    return PSNR, SSIM


def train_SRResnet(epochs):
    device = config.device
    generator = GeneratorResNet().to(device)
    optim = torch.optim.Adam(generator.parameters(), lr=config.model_lr, betas=config.model_betas, eps=config.model_eps,
                             weight_decay=config.model_weight_decay)
    trainLoader = DataLoader(dataset=ImageDataset(mode='train', gt_size=256, root=config.train_root, upscale=False),
                             batch_size=32)
    valLoader = DataLoader(dataset=ImageDataset(mode='val', gt_size=256, root=config.val_root, upscale=False),
                           batch_size=1)
    testLoader = DataLoader(dataset=ImageDataset(mode='test', gt_size=-1, root=config.test_root, upscale=False),
                            batch_size=1)
    cri = nn.L1Loss()
    best_psnr, best_ssim = 0, 0
    lr_scheduler = MultiStepLR(optim, milestones=config.milestones, gamma=config.gamma)
    for epoch in range(epochs):

        epoch_loss = train_srresnet_epoch(generator, optim, cri, trainLoader)
        psnr, ssim = test_srresnet(generator, valLoader)
        lr_scheduler.step()
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim

            torch.save(generator.state_dict(), "pretrain_model/generator.pth")
        wandb.log({'sr_lr': optim.param_groups[0]['lr'],
                   "sr_epoch": epoch,
                   "sr_loss": epoch_loss,
                   'sr_val_psnr': psnr,
                   'sr_best_psnr': best_psnr,
                   'sr_val_ssim': ssim,
                   'sr_best_ssim': best_ssim
                   })
    generator.load_state_dict(torch.load("pretrain_model/generator.pth"))
    test_psnr, test_ssim = test_srresnet(generator, testLoader, save_image=True)
    print(test_psnr, test_ssim)
    print(type(test_psnr), type(test_ssim))
    print("test on Set5: psnr: {:.5f}, ssim: {:.5f}".format(test_psnr, test_ssim))


if __name__ == '__main__':
    wandb.init(project='SRGAN', )
    # train_SRResnet(800)
    train_srgan(80)
