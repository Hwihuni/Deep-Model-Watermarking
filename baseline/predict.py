import argparse
import logging
import os
import scipy.io
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from unet import *
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader

def predict_net(net,path,device):

    net.eval()
    val = BasicDataset(path,acceleration=4,center_fraction=0.08)
    n_val = len(val)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    out = np.zeros((n_val,1,320,320))
    i = 0
    for batch in val_loader:
        imgs = batch['image']
        imgs = imgs.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            mask_pred = net(imgs)

        im_pred = mask_pred.cpu().detach().numpy()
        out[i:i+im_pred.shape[0],:,:,:] = im_pred
        i = i+im_pred.shape[0]
        logging.info(f'Inference done with {i} out of {n_val}')
    logging.info('Inference done')
    
    return out, val.target



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=30,help='Batch size', dest='batch_size')
    parser.add_argument('-f', '--load', dest='load', type=str, default='./checkpoints/LR_1e-05/epoch353.pth',help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,help='Downscaling factor of the images')
    parser.add_argument('--which_noise',choices=("clean","in_mask","out_mask", "every"),default="clean",type=str,help="Type of noise mask")
    parser.add_argument('--noise_std', type=float, default=5e-3,help='Learning rate')
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='2',help='gpu')
    parser.add_argument('-pte', '--path_test', dest='path_test', type=str, default='../data_fastmri/merged_train_2.mat',help='Test dataset')
    parser.add_argument('-ptes', '--path_testsr', dest='path_testsr', type=str, default='../data_fastmri/merged_train_3.mat',help='Test dataset')
    parser.add_argument('-pv', '--path_val', dest='path_val', type=str, default='../data_fastmri/val.mat',help='Val dataset')

    parser.add_argument('-pi', '--path_inf', dest='path_inf', type=str, default='../data_wm',help='Inference destination')
    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ind
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.load, map_location=device))
    logging.info(f'Checkpoint loaded from {args.load}')
    net.to(device=device)
    dir_name = os.path.join(args.path_inf,args.load[14:-4].replace('/','_'))
    os.makedirs(dir_name, exist_ok=True)
    logging.info(f'directory named {dir_name} is made')
    
    img, target  = predict_net(net=net,path = args.path_testsr,device=device)
    savedict = {'recon':np.squeeze(img[:,0,:,:]),'target':np.squeeze(target)}
    filename = dir_name+'/test.mat'
    load = hdf5storage.savemat(filename,savedict, format='7.3')
    logging.info(f'Inference saved in {filename}')

    img, target  = predict_net(net=net,path = args.path_val,device=device)
    savedict = {'recon':np.squeeze(img[:,0,:,:]),'target':np.squeeze(target)}
    filename = dir_name+'/val.mat'
    load = hdf5storage.savemat(filename,savedict, format='7.3')
    logging.info(f'Inference saved in {filename}')
    
    img, target  = predict_net(net=net,path = args.path_test,device=device)
    savedict = {'recon':np.squeeze(img[:,0,:,:]),'target':np.squeeze(target)}
    filename = dir_name+'/train.mat'
    load = hdf5storage.savemat(filename,savedict, format='7.3')
    logging.info(f'Inference saved in {filename}')  
