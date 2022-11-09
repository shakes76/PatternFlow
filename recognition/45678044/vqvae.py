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
    
    # Preload the images to improve the speed of dataloaders
    print('Loading training data')
    train_data = preload_imgs(
        '\data\keras_png_slices_data\keras_png_slices_train')
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=args.batch, 
                                               shuffle=True, num_workers=0)

    print('Loading testing data')
    valid_data = preload_imgs(
        '\data\keras_png_slices_data\keras_png_slices_validate')
    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                               batch_size=args.batch,
                                               shuffle=True, num_workers=0)

    print('Loading validation data')
    test_data = preload_imgs(
        '\data\keras_png_slices_data\keras_png_slices_test')
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=32, 
                                              shuffle=True, num_workers=0)
    
    print('Start to train VQVAE')
    vqvae = VQVAE(1, args.k, args.d).to(args.device)
    optim = torch.optim.Adam(params=vqvae.parameters(), lr=args.lr)
    train_status = train(vqvae, optim, args.epoch, train_loader)

    
        
    print('Start to train PixelCNN')
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=args.batch_prior, 
                                               shuffle=True, num_workers=0)
    pixelcnn = GatedPixelCNN(args.k, 64, 10).to(args.device)
    optim = torch.optim.Adam(params=pixelcnn.parameters(), lr=args.lr_prior)
    train_status = train_prior(vqvae, pixelcnn, optim, args.epoch_prior, train_loader)

    
    print('Generating')
    pixelcnn.eval()
    generated_q = pixelcnn.generate(shape=(64, 64), batch_size=32)
    show_q(generated_q, save_path='/images/generated_q.png')
    show_generated(vqvae, generated_q, save_path='/images/generated_imgs.png')
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VQVAE')
    
    parser.add_argument('--epoch', type=int, default=50,
        help='Epoch size for training vqvae (default: 50)')
    parser.add_argument('--batch', type=int, default=32,
        help='Batch size for training vqvae (default: 32)')
    parser.add_argument('--lr', type=float, default=0.002,
        help='learning rate for training vqvae (default: 0.002)')
    
    parser.add_argument('--epoch_prior', type=int, default=100,
        help='Epoch size for training pixelcnn (default: 100)')
    parser.add_argument('--batch_prior', type=int, default=64,
        help='Batch size for training pixelcnn (default: 64)')
    parser.add_argument('--lr_prior', type=float, default=0.001,
        help='learning rate for training pixelcnn (default: 0.001)')
    
    parser.add_argument('--k', type=float, default=512,
        help='Num of latent vectors (default: 512)')
    parser.add_argument('--d', type=float, default=64,
        help='Dim of latent vectors (default: 64)')
    
    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    main(args)
    
