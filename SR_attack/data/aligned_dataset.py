import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import hdf5storage
import numpy as np

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        # super(AlignedDataset,self).__init__(opt)
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        load = hdf5storage.loadmat(self.dir_AB)
        self.target = load['target']
        self.recon = load['recon']
        self.mean_signal =np.mean(self.target)
        num_cols = self.target.shape[-2]
        num_low_freqs = int(round(num_cols * opt.center_fraction))

        # create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (opt.acceleration * (num_low_freqs - num_cols)) / (
            num_low_freqs * opt.acceleration - num_cols
        )
        
        offset = 0
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True
        
        # reshape the mask
        mask_shape = [1,320,1]
        mask = mask.reshape(*mask_shape)
        self.mask = np.repeat(mask,320,axis =2)

    @classmethod
    def undersample(cls, x,mask):
        x_k = fft2c(x)
        masked_kspace_sudo = mask*x_k +0.0
        image_input_sudo = np.abs(ifft2c(masked_kspace_sudo))
        return image_input_sudo, x

    def __getitem__(self, index):
        AB_path = self.dir_AB
        A, _ = self.undersample(self.target[index:index+1,:,:],self.mask)
        B = self.recon[index:index+1,:,:]
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.dir_AB)

    def name(self):
        return 'AlignedDataset'

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1))),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1))),axes = (-2,-1))