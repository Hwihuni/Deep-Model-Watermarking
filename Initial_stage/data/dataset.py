
import logging

import hdf5storage
import numpy as np
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self,path,acceleration,center_fraction):
        load = hdf5storage.loadmat(path)
        self.target = load['target']
        self.recon = load['recon']
        
        self.mean_signal =np.mean(self.target)
        num_cols = self.target.shape[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
            num_low_freqs * acceleration - num_cols
        )
        
        offset = 0
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True
        
        # reshape the mask
        mask_shape = [1,320,1]
        mask = mask.reshape(*mask_shape)
        self.mask = np.repeat(mask,320,axis =2)

        logging.info(f'Creating dataset with {self.target.shape[0]} examples')
        
    @classmethod
    def undersample(cls, x,mask):
        x_k = fft2c(x)
        masked_kspace_sudo = mask*x_k +0.0
        image_input_sudo = np.abs(ifft2c(masked_kspace_sudo))
        return image_input_sudo, x

    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, i):
        img, _ = self.undersample(self.target[i:i+1,:,:],self.mask)
        return np.concatenate((img,self.recon[i:i+1,:,:]),axis = 2)
        '''
        return {
            'image': img, 'target': target,'recon': self.recon[i:i+1,:,:]
        }
        '''

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1))),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1))),axes = (-2,-1))

def null_space_prof(input,target,power = 1):
    output = input-target*np.sum(target*input)/np.sum(target**2)
    output = power*output/np.sqrt(np.mean(output**2))
    return output
