
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Constants
epoch_range = 1

batch_size=120
train_factor=2#00
valid_factor=1#0
test_factor=10#0

modulo=round(train_factor/10) +1#Print frequency while training

#Importing Custom Dataloader
import dataset #as data
train_loader, valid_loader, test_loader, clas_dataset =dataset.dataset(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor)

#Importing CNN Model
import modules

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = modules.Net()
net = net.to(device)

# No backpropagtion , No need for calculating gradients, => Faster calculation
with torch.no_grad():
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):

        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        slice_number = data[2].to(device) 

        class_image_NC, class_image_AD = dataset.clas_output(clas_dataset,slice_number,inputs)

        output1,output2 = net(inputs,class_image_NC)#.squeeze(1)
        euclidean_distance_NC = F.pairwise_distance(output1, output2)

        output1,output2 = net(inputs,class_image_AD)#.squeeze(1)
        euclidean_distance_AD = F.pairwise_distance(output1, output2)

        euclidean_distance_MAX = torch.ge(euclidean_distance_AD,euclidean_distance_NC)
        
        correct_tensor=torch.eq(labels,euclidean_distance_MAX)

        correct_run = torch.sum(correct_tensor)
        correct += correct_run
