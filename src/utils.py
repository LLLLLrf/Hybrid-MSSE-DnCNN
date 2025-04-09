import math
import torch
import torch.nn as nn
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def batch_PSNR(output, target, data_range):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    total_psnr = 0.0
    for i in range(output.shape[0]):
        total_psnr += compare_psnr(target[i], output[i], data_range=data_range)
    return total_psnr / output.shape[0]


def batch_SSIM(output, target, data_range, win_size=7):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    total_ssim = 0.0
    for i in range(output.shape[0]):
        min_side = min(target[i].shape[:2])
        adjusted_win_size = min(win_size, min_side if min_side % 2 == 1 else min_side - 1)
        if min_side >= adjusted_win_size:
            total_ssim += compare_ssim(target[i], output[i], data_range=data_range, multichannel=True, win_size=adjusted_win_size, use_sample_covariance=False)
        else:
            raise ValueError(f"Image size is too small for the given win_size. Image size: {target[i].shape}, win_size: {adjusted_win_size}")
    return total_ssim / output.shape[0]


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)


def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
