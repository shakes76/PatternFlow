
#Defining CNN

import torch
import torch.nn as nn
import torch.nn.functional as F

#Solver
import torch.optim as optim

#
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np

#Garbage collector
import gc 

#Importing CNN Model
import modules


#Scheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = modules.Net_3D()
#net.load_state_dict(torch.load('sim_net_ResNet.pt')) #change to .pt
#net.eval()
net = net.to(device)


#torch.save(net.state_dict(), 'sim_net_ResNet.pt')

#Constants
epoch_range = 5#00


batch_size=20*3
train_factor=10
test_factor=40
valid_factor=10

modulo=round(train_factor*20/(batch_size*10))+1 #Print frequency while training

#Importing Custom Dataloader
import dataset 

#Preparation
    
train_image_paths, valid_image_paths = dataset.train_valid_data(TRAIN_DATA_SIZE = 4*20,VALID_DATA_SIZE = 20)
train_dataset = dataset.DatasetTrain3D(train_image_paths,dataset.transformation_3D())
train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=1)


for i, data in enumerate(train_loader, 0):
    inputs_1= data[0].to(device)

    print(inputs_1.shape)

    inputs_1= data[1].to(device)

    output=net.forward_once(inputs_1)

    print(output.shape)
    break
    
    break