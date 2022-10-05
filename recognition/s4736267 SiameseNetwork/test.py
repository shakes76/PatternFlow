
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

batch_size=128
train_factor=2#00
valid_factor=1#0
test_factor=1#0

modulo=round(train_factor/10) +1#Print frequency while training

#Importing Custom Dataloader
import dataset as data
train_loader, valid_loader, test_loader =data.dataset(batch_size,TRAIN_SIZE = train_factor*20, VALID_SIZE= valid_factor*20, TEST_SIZE=test_factor*20)

data.classification_data()


for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    print(f'EPOCH NUMBER: {epoch} =', end ="", flush=True) 


    total = 0

    for i, data in enumerate(train_loader, 0):
        

        inputs_1= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        #print(inputs_1)
        
        break
    
####


