from models import VQVAE
from helper import train
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     ])


train_data = datasets.MNIST(
    root = './data',
    train = True,                         
    transform = transform, 
    download = True,            
)

test_data = datasets.MNIST(
    root = './data', 
    train = False, 
    transform = transform,
    download = True,  
)

EPOCH_SIZE = 500
BATCH_SIZE = 128
LR = 0.00001
K = 512
D = 64

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, 
                                           shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, 
                                          shuffle=True, num_workers=0)

vqvae = VQVAE(1, K, D)
optim = torch.optim.Adam(params=vqvae.parameters(), lr=LR)
train(vqvae, optim, EPOCH_SIZE, train_loader)


