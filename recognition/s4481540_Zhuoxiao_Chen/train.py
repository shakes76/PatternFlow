import argparse
import random
import math

from data_loader import Dataset
from network import Styled_G, D

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from PIL import Image

def stacking_parameters(net_0, net_1, weight_decay=0.999):
    """Accumulate the parameters of two models based on the weight decay"""
    parameter_0, parameter_1 = dict(net_0.named_parameters()),\
                               dict(net_1.named_parameters())

    for key in parameter_0.keys():
        parameter_0[key].data.mul_(weight_decay).add_(1 - weight_decay,
                                                    parameter_1[key].data)

if __name__ == '__main__':
    batch_size, latent_length, n_critic = 16, 512, 1

    parser = argparse.ArgumentParser(description='StyleGAN')

    parser.add_argument('path', type=str, help='Dataset Path ')
    parser.add_argument('--phase', type=int, default=40_000,
                        help='number of samples for each training phase')
    parser.add_argument('--init_size', default=8, type=int,
                        help='initial size of input images')
    parser.add_argument('--sched', action='store_true',
                        help='scheduling for lr')
    parser.add_argument('--max_size', default=512, type=int,
                        help='max size of generated images')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoints')
    parser.add_argument( '--no_from_rgb_activate', action='store_true',
                         help='activate in from_rgb')
    parser.add_argument( '--mixing', action='store_true',
                         help='mixing regularization')
    parser.add_argument( '--loss', type=str, default='wgan-gp',
                         choices=['wgan-gp', 'r1'], help='choose gan loss')
    args = parser.parse_args()

    G_net = nn.DataParallel(Styled_G(latent_length)).cuda()
    D_net = nn.DataParallel(
        D(from_rgb_activate=not args.no_from_rgb_activate)).cuda()
    G_processing = Styled_G(latent_length).cuda()
    G_processing.train(False)

    beta_0 = 0.0
    beta_1 = 0.99
    G_optimiser = optim.Adam(
        G_net.module.generator.parameters(), lr=args.lr, betas=(beta_0, beta_1)
    )
    G_optimiser.add_param_group(
        {
            'params': G_net.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    D_optimiser = optim.Adam(D_net.parameters(), lr=args.lr,
                             betas=(beta_0, beta_1))