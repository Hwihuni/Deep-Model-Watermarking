import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import scipy.io
from eval import eval_net
from unet import UNet
from util import *
from loss import *
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader


dir_checkpoint = '/home/hwihun/Deep-Model-Watermarking/baseline/checkpoints/'

def train_net(net,device,args):

    train = BasicDataset(args.path_train,acceleration=4,center_fraction=0.08)
    val = BasicDataset(args.path_val,acceleration=4,center_fraction=0.08)
    n_train = len(train)
    n_val = len(val)
    

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.L1Loss()
    #test_loss = []
    #val_loss =[]
    for epoch in range(args.epochs):
        net.train()
        
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['target']
                if torch.cuda.device_count() > 1:
                    imgs = imgs.to(device='cuda:1', dtype=torch.float32)
                    true_masks = true_masks.to(device='cuda:1', dtype=torch.float32)
                else:
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                masks_pred = net(imgs)
               
                loss_l1 = criterion(masks_pred, true_masks)
                if torch.cuda.device_count() > 1:
                    loss_grad = grad_loss(masks_pred, true_masks,device='cuda:1')
                else:
                    loss_grad = grad_loss(masks_pred, true_masks,device=device)
                loss = loss_l1+0.1*loss_grad
               # loss = loss_l1
                epoch_loss += loss.item()
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train-L1', loss_l1.item(), global_step)
                writer.add_scalar('Loss/train-grad', loss_grad.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_value_(net.parameters(), 1)
                optimizer.step()                               
                pbar.update(imgs.shape[0])
                
                global_step += 1
                
            if epoch % args.valid_step == 0:
                if torch.cuda.device_count() > 1:
                    val_score_l1, val_score_grad = eval_net(net, val_loader,device='cuda:1', writer=writer,e=epoch)
                else:
                    val_score_l1, val_score_grad = eval_net(net, val_loader,device=device, writer=writer,e=epoch)
                logging.info('L1 Loss: {}'.format(val_score_l1))
                logging.info('Gradient Loss: {}'.format(val_score_grad))
                writer.add_scalar('Loss/val', val_score_l1+0.1*val_score_grad, epoch)
                writer.add_scalar('Loss/val-L1', val_score_l1, epoch)
                writer.add_scalar('Loss/val-Grad', val_score_grad, epoch)

                #val_loss =np.append(val_loss,val_score_l1+0.1*val_score_grad)
                #test_loss =np.append(test_loss,test_score_l1+0.1*test_score_grad)
                #np.save('inf/val_loss', val_loss)
                #np.save('inf/test_loss', test_loss)
                    
            
        if epoch % args.save_step == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            os.makedirs(dir_checkpoint + f'LR_{args.lr}', exist_ok=True)
            torch.save(net.state_dict(),dir_checkpoint + f'LR_{args.lr}/epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,help='Learning rate', dest='lr')
    parser.add_argument('-lo', '--load', dest='load', type=str, default=False,help='Load model from a .pth file')
    parser.add_argument('-pv', '--path_val', dest='path_val', type=str, default='/home/hwihun/Deep-Model-Watermarking/data/val.mat',help='Validation dataset') 
    parser.add_argument('-pt', '--path_train', dest='path_train', type=str, default='/home/hwihun/Deep-Model-Watermarking/data/merged_train_1.mat',help='training dataset')
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='0',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=1,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=1,help='Checkpoint saving step')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ind
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    net.apply(init_weights)
    net.to(device=device)

    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')
    
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, output_device=1)
        logging.info(f'Using multi-GPU')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
        

    train_net(net=net,device=device,args = args)
