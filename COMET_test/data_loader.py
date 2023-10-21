import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.ndimage
from torch.nn.functional import interpolate
from random import randint


class LoadData(Dataset):
    def __init__(self, path,path_HR, path_LR,type, slc,down_rate=2, patch_size=32, times=8):
        # num img_H img_W 31
        self.zong = sio.loadmat(path)

        
        self.down_rate = down_rate
        self.patch_size = patch_size 
        self.slc = slc
        self.times = times

        self.hr_hsi = np.load(path_HR).astype(np.float32)
        self.lr_hsi = np.load(path_LR).astype(np.float32)
        # self.lr_hsi_k = np.load(path_k_hole).astype(np.float32)

        self.S0 = np.array(self.zong['S0']).astype(np.float32)
        self.mask = np.array(self.zong['mask']).astype(np.float32)

        self.shape = self.hr_hsi.shape

        self.type = type


        self.hr_msi = self._scl()
        self.hr_hsi = torch.from_numpy(self.hr_hsi)

        # self.lr_hsi = torch.from_numpy(self.lr_hsi)
        self.lr_hsi = torch.from_numpy(self.lr_hsi)
        # self.lr_hsi_k = torch.from_numpy(self.lr_hsi_k)


        self.S0 = torch.from_numpy(self.S0)
        self.mask = torch.from_numpy(self.mask)
        self.hr_msi = torch.from_numpy(self.hr_msi)
        self.hr_hsi = self.hr_hsi.type(torch.FloatTensor)
        self.lr_hsi = self.lr_hsi.type(torch.FloatTensor)
        # self.lr_hsi_k = self.lr_hsi_k.type(torch.FloatTensor)
        self.hr_msi = self.hr_msi.type(torch.FloatTensor)
        self.S0 = self.S0.type(torch.FloatTensor)
        self.mask = self.mask.type(torch.FloatTensor)

    def _down_sample(self):
        # return scipy.ndimage.zoom(self.hr_hsi, zoom=[1.0, 1.0, 1.0 / self.down_rate, 1.0/self.down_rate], order=0)  # nearest临阶插值  
        return scipy.ndimage.zoom(self.hr_hsi, zoom=[1.0, 1.0, 1.0 / self.down_rate, 1.0/self.down_rate], order=3)
        

    def make_patch(self):
        a = np.zeros(shape=(self.times*self.shape[0],self.shape[1],self.patch_size,self.patch_size))
        for i in range(self.shape[0]):
            h = np.random.randint(low=int(self.patch_size/2)+1, high=self.shape[2]-int(self.patch_size/2)-1, size=self.times, dtype=int)
            w = np.random.randint(low=int(self.patch_size/2)+1, high=self.shape[3]-int(self.patch_size/2)-1,size=self.times, dtype=int)
            for j in range(self.times):
                b = np.expand_dims(self.hyperHR[i, :, (h[j]-int(self.patch_size/2)):(h[j]+int(self.patch_size/2)), (w[j]-int(self.patch_size/2)):(w[j]+int(self.patch_size/2))],axis=0)
                a[j+i*self.times,:,:,:] = a[j+i*self.times,:,:,:] + b
        return a

    def _scl(self):
        if self.type != 'test':
            a = np.zeros((self.times*self.shape[0], len(self.slc), self.hr_hsi.shape[2], self.hr_hsi.shape[3]))
            for i in range(self.times*self.shape[0]):
                for index, j in enumerate(self.slc):
                    a[i,index,:,:] = self.hr_hsi[i,int(j),:,:]
        else:
            a = np.zeros((self.shape[0], len(self.slc), self.hr_hsi.shape[2], self.hr_hsi.shape[3]))
            for i in range(self.shape[0]):
                for index, j in enumerate(self.slc):
                    a[i,index,:,:] = self.hr_hsi[i,int(j),:,:]
        return a

    def __getitem__(self, index):
        return self.lr_hsi[index,:,:,:], self.hr_msi[index,:,:,:], self.hr_hsi[index,:,:,:], self.mask[index,:,:], self.S0[index,:,:]

        # return self.lr_hsi[index, :, :, :] / torch.max(self.lr_hsi[index, :, :, :]), self.hr_msi[index, :, :,
        #                                                                              :], self.hr_hsi[index, :, :,
        #                                                                                  :], self.mask[index, :,
        #                                                                                      :], self.S0[index, :, :]
    def __len__(self):
        return self.hr_hsi.shape[0]