import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import scipy.ndimage
import collections
from matplotlib import pyplot as plt

def loadData(mname):
    return sio.loadmat(mname)



def adjust_learning_rate(opti, rate_decay):
    lr = opti.param_groups[0]['lr'] * rate_decay
    for param_group in opti.param_groups:
        param_group['lr'] = lr


def show_learning_rate(opti):
    return opti.param_groups[0]['lr']


def save_logger(file, file_path):
    with open(file_path, 'a+') as f:
        f.write(file + '\n')


def load_data(path, str_name, dev):
    data = sio.loadmat(path)[str_name].astype(np.float32)
    data = torch.from_numpy(data).to(dev)
    # data = torch.unsqueeze(data, 0)
    return data

def calc_psnr(img1, img2):
    return 10. * torch.log10((255.0 **2)/ torch.mean((img1 - img2) ** 2))

def PSNR_GPU(img1, img2):
    mpsnr = 0
    for l in range(img1.size()[1]):

        mpsnr += 10. * torch.log10(1. / torch.mean((img1[:,l,:,:] - img2[:,l,:,:]) ** 2))

    return mpsnr / img1.size()[1]

    # return 10. * torch.log10((torch.max(img1)**2) / torch.mean((img1 - img2) ** 2))

def PSNR_GPU_mask(pred, gt, mask):
    mpsnr = 0
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    for r in range(pred.shape[0]):
        temp_mask = mask[r,:,:,:]
        temp_gt = gt[r, :,:,:]
        temp_pred = pred[r,:,:,:]

        mpsnr += 10.*np.log10(
            1./np.mean((temp_pred[np.where(temp_mask != 0)] - temp_gt[np.where(temp_mask != 0)]) ** 2)
        )
    return mpsnr / pred.shape[0]


def SAM(pred, gt):
    pred = pred.numpy()
    gt = gt.numpy()
    eps = 2.2204e-16
    pred[np.where(pred == 0)] = eps
    gt[np.where(gt == 0)] = eps

    nom = sum(pred * gt)
    denom1 = sum(pred * pred) ** 0.5
    denom2 = sum(gt * gt) ** 0.5
    sam = np.real(np.arccos(nom.astype(np.float32) / (denom1 * denom2 + eps)))
    sam[np.isnan(sam)] = 0
    sam_sum = np.mean(sam) * 180 / np.pi
    return sam_sum


def plot(logger_path, save_map_path):   # 这里是从val的logger中得到对应的数据。

    psnr = []
    sam = []

    with open(save_map_path, 'r') as f:

        for line in f.readlines():

            line = line.strip()

            if 'val averagr psnr : ' in line:
                line = line.split(' ')
                psnr.append(float(line[-4]))
                sam.append(float(line[-1]))

    epochs = [i for i in range(len(psnr))]

    fib_size = (5, 4)
    fon_size = 12

    plt.figure(figsize=fib_size)
    plt.title('sam of every epoch', fontsize=fon_size)
    plt.xlabel('epoch', fontsize=fon_size)
    plt.ylabel('sam', fontsize=fon_size)
    plt.plot(epochs, sam, 'k.')
    plt.grid(True, linestyle="-.", color="k", linewidth="1.1")
    plt.savefig(save_map_path.joinpath('icvl_sam.png'))

    plt.figure(figsize=fib_size)
    plt.title('psnr of every epoch', fontsize=fon_size)
    plt.xlabel('epoch', fontsize=fon_size)
    plt.ylabel('psnr', fontsize=fon_size)
    plt.plot(epochs, psnr, 'k.')
    plt.grid(True, linestyle="-.", color="k", linewidth="1.1")
    plt.savefig(save_map_path.joinpath('icvl_psnr.png'))

def save_mat(root_path, hr_hsi, epoch):

    img = np.array(hr_hsi[0].cpu().numpy())
    if len(hr_hsi) != 1:
        for i in range(1,len(hr_hsi)):
            img = np.concatenate((img, np.array(hr_hsi[i].cpu().numpy())),axis=0)

    # hr_hsi = hr_hsi.cpu().numpy()
    # hr_hsi = np.array(hr_hsi)
    data = {'hr_hsi': img}
    # hr_msi = pred_hr_msi.detach().cpu().numpy()
    # hr_msi = np.array(hr_msi)
    # data['hr_msi'] = hr_msi
    # hr_hsi = pred_hr_hsi.detach().cpu().numpy()
    # hr_hsi = np.array(hr_hsi)
    # data['hr_hsi'] = hr_hsi
    sio.savemat(os.path.join(root_path, str(epoch) + '.mat'), data)
    return os.path.join(root_path, str(epoch) + '.mat')