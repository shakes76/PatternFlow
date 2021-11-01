from modules import *
from helper import *
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
    print('Loading training data')
    train_data = preload_imgs(
        '\data\keras_png_slices_data\keras_png_slices_train')
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=args.batch, 
                                               shuffle=True, num_workers=0)
    print('Finished')
    print('Loading testing data')
    valid_data = preload_imgs(
        '\data\keras_png_slices_data\keras_png_slices_validate')
    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                               batch_size=args.batch,
                                               shuffle=True, num_workers=0)
    print('Finished')
    print('Loading validation data')
    test_data = preload_imgs(
        '\data\keras_png_slices_data\keras_png_slices_test')
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=32, 
                                              shuffle=True, num_workers=0)
    print('Finished')
    
    print('Start to train VQVAE')
    vqvae = VQVAE(1, args.k, args.d).to(args.device)
    optim = torch.optim.Adam(params=vqvae.parameters(), lr=args.lr)
    train_status = train(vqvae, optim, args.epoch, train_loader)
    print('Finished')
    
    
    print('Start to train PixelCNN prior')
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VQVAE')
    
    parser.add_argument('--epoch', type=int, default=50,
        help='Epoch size for training vqvae (default: 50)')
    parser.add_argument('--batch', type=int, default=32,
        help='Batch size for training vqvae (default: 32)')
    parser.add_argument('--lr', type=float, default=0.002,
        help='learning rate for training vqvae (default: 0.002)')
    
    parser.add_argument('--k', type=float, default=512,
        help='Num of latent vectors (default: 512)')
    parser.add_argument('--d', type=float, default=64,
        help='Dim of latent vectors (default: 64)')
    
    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    main(args)
    
