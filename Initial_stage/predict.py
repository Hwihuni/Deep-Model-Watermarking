# encoding: utf-8


import argparse
import os
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import utils.transformed as transforms
from data.dataset import BasicDataset
from models.HidingUNet import UnetGenerator

import numpy as np
from PIL import Image
import hdf5storage

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",help='train | val | test')
parser.add_argument('--workers', type=int, default=4,help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=40,help='input batch size')
parser.add_argument('--imageSize', type=int, default=320,help='the number of frames')
parser.add_argument('--experiment_dir',default='../HR/n10__2022-02-03-11_03_03',help='the number of frames')
parser.add_argument('--cuda', type=bool, default=True,help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,help='number of GPUs to use')
parser.add_argument('--Hnet', default='netH_epoch_4,sumloss=87.167397,Hloss=86.507820.pth',help="path to Hidingnet (to continue training)")
parser.add_argument('--trainpics', default='../HR/',help='folder to output training images')

#datasets to train
parser.add_argument('--datasets', type=str, default='../data_wm/LR_1e-05_epoch353/',help='denoise/derain')
parser.add_argument('--num_downs', type=int, default= 1 , help='nums of  Unet downsample')
parser.add_argument('--gpu_ind', type=str, default='2',help='gpu')


def main():
    ############### define global parameters ###############
    global opt
    global test_loader, test_dataset
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ind
    if torch.cuda.is_available() and not opt.cuda:
        logging.info("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create the dirs to save the result #############

    experiment_dir = opt.experiment_dir
    opt.outckpts = experiment_dir + "/checkPoints"
    opt.testPics = experiment_dir + "/testPics"

    if (not os.path.exists(opt.testPics)):
        os.makedirs(opt.testPics)
    Hnet = UnetGenerator(input_nc=2, output_nc=1, num_downs= opt.num_downs, output_function=nn.Sigmoid)
    Hnet.cuda()

    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.outckpts + '/' + opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    logging.info(f'Hnet is loaded with {opt.outckpts}/{opt.Hnet}')

    DATA_DIR = opt.datasets
    testdir = os.path.join(DATA_DIR, 'test.mat')
        
    test_dataset = BasicDataset(testdir,4,0.08)

    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                            shuffle=False, num_workers=int(opt.workers))    	


    
    img = test(test_loader, Hnet=Hnet)
    savedict = {'recon_wm':np.squeeze(img),'target':np.squeeze(test_dataset.target)}
    filename = opt.testPics +'/test_wm.mat'
    hdf5storage.savemat(filename,savedict, format='7.3')
    logging.info(f'Inference saved in {filename}')



def test(test_loader,  Hnet):
    logging.info(
        "#################################################### test begin ########################################################")
    Hnet.eval()

    # Tensor type
    Tensor = torch.cuda.FloatTensor 

    loader = transforms.Compose([trans.Resize((320,320)),trans.Grayscale(num_output_channels=1),transforms.ToTensor()])
    secret_img = Image.open("../secret/BIT.png")
    secret_img = loader(secret_img)  
    out = np.zeros((len(test_dataset),320,320))
    i = 0
    for _, data in enumerate(test_loader, 0):

        this_batch_size = int(data.size()[0])  
        cover_img_B = data[ 0:this_batch_size, :, 0:320, 320:640]

        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
        secret_img = secret_img[0:this_batch_size, :, :, :]  

        if opt.cuda:
            cover_img_B = cover_img_B.type(torch.FloatTensor).cuda()        
            secret_img = secret_img.cuda()

        concat_img = torch.cat([cover_img_B, secret_img], dim=1)
        concat_imgv = Variable(concat_img)  
        container_img = Hnet(concat_imgv)  
        out[i:i+this_batch_size,:,:] = np.squeeze(container_img.cpu().detach().numpy())
        i = i+this_batch_size
        logging.info(f'Inference done with {i} out of {len(test_dataset)}')
    logging.info("#################################################### test end ########################################################")

    return out



if __name__ == '__main__':
    main()
