import h5py
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
from skimage.transform import resize
import hdf5storage

def null_space_prof(input,target,power = 1):
    output = input-target*np.sum(target*input)/np.sum(target**2)
    output = power*output/np.sqrt(np.mean(output**2))
    return output

category = 'ssim'
mask = True
if category == 'ssim':
    from skimage.metrics import structural_similarity as func
elif category == 'nrmse':
    from skimage.metrics import normalized_root_mse as func
elif category == 'psnr':
    from skimage.metrics import peak_signal_noise_ratio
    def func(x,y):
        return peak_signal_noise_ratio(x,y,data_range=1.5)

SNR = 10
shape = (1,320,320)

noise_gauss_img = np.random.randn(shape[0],shape[1],shape[2])
noise_rice_img = np.random.rayleigh(size=(shape[0],shape[1],shape[2]))
noise_snp_img = np.round(np.random.rand(shape[0],shape[1],shape[2]))
noise_snp_zeromean_img = np.round(np.random.rand(shape[0],shape[1],shape[2]))*2-1

tmp = np.round(np.random.rand(shape[0],shape[1],shape[2]))
noise_snp_rec_img = np.zeros_like(tmp)
noise_snp_rec_img[:,150:170,150:170] = tmp[:,150:170,150:170]

value_gauss = 0
value_rice = 0
value_snp_zeromean = 0
value_snp_rec = 0
value_snp = 0
for i in range(400,4800,400):
    
    load = hdf5storage.loadmat(f'./data/train{i}.mat')
    


    target = load['target']
    mean_signal = np.mean(target)
    noise_std = mean_signal/(np.sqrt(np.mean(noise_gauss_img**2))*SNR)
    image_gauss_img = np.abs(target+noise_std*noise_gauss_img)
    noise_std = mean_signal/(np.sqrt(np.mean(noise_rice_img**2))*SNR)
    image_rice_img = np.abs(target+noise_std*noise_rice_img)
    noise_std = mean_signal/(np.sqrt(np.mean(noise_snp_img**2))*SNR)
    image_snp_img = np.abs(target+noise_std*noise_snp_img)
    noise_std = mean_signal/(np.sqrt(np.mean(noise_snp_zeromean_img**2))*SNR)
    image_snp_zeromean_img = np.abs(target+noise_std*noise_snp_zeromean_img)
    noise_std = mean_signal/(np.sqrt(np.mean(noise_snp_rec_img**2))*SNR)
    image_snp_rec_img = np.abs(target+noise_std*noise_snp_rec_img)
    if mask:
        image_gauss_img = np.where(target>0.1,image_gauss_img,target)
        image_rice_img = np.where(target>0.1,image_rice_img,target)
        image_snp_img = np.where(target>0.1,image_snp_img,target)
        image_snp_zeromean_img = np.where(target>0.1,image_snp_zeromean_img,target)
        image_snp_rec_img = np.where(target>0.1,image_snp_rec_img,target)
    slice = target.shape[0]
    value_gauss += func(target[0:int(slice/2),:,:],image_gauss_img[0:int(slice/2),:,:])
    value_gauss += func(target[int(slice/2):slice,:,:],image_gauss_img[int(slice/2):slice,:,:])

    value_rice += func(target[0:int(slice/2),:,:],image_rice_img[0:int(slice/2),:,:])
    value_rice += func(target[int(slice/2):slice,:,:],image_rice_img[int(slice/2):slice,:,:])

    
    value_snp += func(target[0:int(slice/2),:,:],image_snp_img[0:int(slice/2),:,:])
    value_snp += func(target[int(slice/2):slice,:,:],image_snp_img[int(slice/2):slice,:,:])
    
    value_snp_zeromean += func(target[0:int(slice/2),:,:],image_snp_zeromean_img[0:int(slice/2),:,:])
    value_snp_zeromean += func(target[int(slice/2):slice,:,:],image_snp_zeromean_img[int(slice/2):slice,:,:])

    value_snp_rec += func(target[0:int(slice/2),:,:],image_snp_rec_img[0:int(slice/2),:,:])
    value_snp_rec += func(target[int(slice/2):slice,:,:],image_snp_rec_img[int(slice/2):slice,:,:])
    div = i/200
    print(i,value_gauss/div,value_rice/div,value_snp/div,value_snp_zeromean/div,value_snp_rec/div)
nrmse_gauss = value_gauss/22
nrmse_rice  = value_rice/22
nrmse_snp_zeromean  = value_snp_zeromean/22
nrmse_snp_rec  = value_snp_rec/22
nrmse_snp = value_snp/22
if mask:
    sio.savemat(f'metric_input/masked_metric_{category}_input_{SNR}.mat',{f'{category}_gauss': nrmse_gauss,f'{category}_rice': nrmse_rice,f'{category}_snp': nrmse_snp,f'{category}_snp_zeromean': nrmse_snp_zeromean,f'{category}_snp_rec': nrmse_snp_rec})
else:
    sio.savemat(f'metric_input/metric_{category}_input_{SNR}.mat',{f'{category}_gauss': nrmse_gauss,f'{category}_rice': nrmse_rice,f'{category}_snp': nrmse_snp,f'{category}_snp_zeromean': nrmse_snp_zeromean,f'{category}_snp_rec': nrmse_snp_rec})


