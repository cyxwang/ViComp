'''
Useful helper functions

collected from https://github.com/BingyaoHuang/CompenNeSt-plusplus

'''

import os
from os.path import join as fullfile
import numpy as np
import cv2 as cv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from compensation import pytorch_ssim
# import lpips
from torch.utils.data import DataLoader
from compensation.differential_color_function import rgb2lab_diff, ciede2000_diff
import lpips



# loss functions
l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda()
lpips_fun = lpips.LPIPS(net='alex').cuda()


# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Same as np.repeat, while torch.repeat works as np.tile
def repeat_np(a, repeats, dim):
    '''
    Substitute for numpy's repeat function. Source from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    '''

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(
            torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)


# save 4D np.ndarray or torch tensor to image files
def saveImgs(inputData, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(i + 1)
        cv.imwrite(fullfile(dir, file_name), imgs[i, :, :, :])  # faster than PIL or scipy


def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

# save 4D np.ndarray or torch tensor to image files
def convertImgs(inputData):
    

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    
    output = imgs[0,:,:,:]
    

    return output
        



# compute PSNR
def psnr(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return 10 * math.log10(1 / l2_fun(x, y))


# compute RMSE
def rmse(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y


    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return math.sqrt(l2_fun(x, y).item() * 3) # only works for RGB, for grayscale, don't multiply by 3


# compute SSIM
def ssim(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        return pytorch_ssim.ssim(x, y).item()


# compute psnr, rmse and ssim
def computeMetrics(x, y):
    l2_fun = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    last_loc = 0
    metric_mse, metric_ssim = 0., 0.
    metric_diff = 0.

    num_imgs = x.shape[0]
    batch_size = 1 


    with torch.no_grad():
        for i in range(0, num_imgs // batch_size):
            idx = range(last_loc, last_loc + batch_size)
            x_batch = x[idx, :, :, :].to(device) if x.device.type != 'cuda' else x[idx, :, :, :]
            y_batch = y[idx, :, :, :].to(device) if y.device.type != 'cuda' else y[idx, :, :, :]
            
            # differential color
            xl_batch = rgb2lab_diff(x_batch,device)            
            yl_batch = rgb2lab_diff(y_batch,device)
            diff_map = ciede2000_diff(xl_batch,yl_batch,device)
            
            color_loss=diff_map.mean()
            metric_diff += color_loss*x_batch.shape[0]

            # compute mse and ssim
            metric_mse += l2_fun(x_batch, y_batch).item() * batch_size
            metric_ssim += ssim(x_batch, y_batch) * batch_size

            last_loc += batch_size

        # average
        metric_mse /= num_imgs
        metric_ssim /= num_imgs        
        metric_diff /= num_imgs

        # rmse and psnr
        metric_rmse = math.sqrt(metric_mse * 3)  # 3 channel image
        metric_psnr = 10 * math.log10(1 / metric_mse)

        
        
        
     

    return metric_psnr, metric_rmse, metric_ssim,metric_diff


# compute loss between prediction and ground truth
def computeLoss(prj_pred, prj_train, loss_option):
    train_loss = 0
    

    # l1
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)
        train_loss = train_loss+ l1_loss

    # l2
    l2_loss = l2_fun(prj_pred, prj_train)
    if 'l2' in loss_option:
        train_loss = train_loss+ l2_loss

    # ssim

    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))
        train_loss = train_loss+ ssim_loss

    if 'lpips' in loss_option:
        lpips_loss = lpips_fun(prj_pred, prj_train)
        train_loss = train_loss+ lpips_loss

    if 'diff' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))
        train_loss = train_loss+ ssim_loss

            # differential color
        xl_batch = rgb2lab_diff(prj_pred,'cuda')            
        yl_batch = rgb2lab_diff(prj_train,'cuda')
        diff_map = ciede2000_diff(xl_batch,yl_batch,'cuda')
            
        color_loss=diff_map.mean()
        train_loss = train_loss + color_loss
        
    return train_loss



# count the number of parameters of a model
def countParameters(model):
    return sum([param.numel() for param in model.parameters() if param.requires_grad])


# generate training title string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['data_name'], train_option['model_name'], train_option['loss'],
                                                  train_option['num_train'], train_option['batch_size'], train_option['max_iters'],
                                                  train_option['lr_cmp'], train_option['lr_warp'],train_option['lr_drop_ratio'], train_option['lr_drop_rate'],
                                                  train_option['l2_reg'])

