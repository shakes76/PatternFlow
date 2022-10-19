
#Defining CNN

#######################################################
#                  Packages
#######################################################

#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
#Math
import math
#Plot
import matplotlib.pyplot as plt
import numpy as np
#Garbage collector
import gc 

#######################################################
#                  Importing custom files
#######################################################

import modules
import dataset 

#######################################################
#                  Defining constants
#######################################################

epoch_range = 200
batch_size=16
train_factor=1000               #Number of Persons
test_factor=400
valid_factor=400

FILE="weights_only_l3.pth"         #File location of saved pretrained net

#(0=disabled)
load_pretrained_model=1
plot_feature_vectors=1          #Ploting feature vectors
plot_loss = 1                   #Ploting training and validation loss


#######################################################
# Loading Net and defining optimizer, scheduler and weight initialisation
#######################################################

#Garbage collection
gc.collect()
torch.cuda.empty_cache()

#Importing CNN Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if load_pretrained_model==0: 
    net = modules.ResNet_3D(modules.Residual_Identity_Block_R3D,modules.Residual_Conv_Block_R3D)
    net = net.to(device)
else:
    
    net = modules.ResNet_3D(modules.Residual_Identity_Block_R3D,modules.Residual_Conv_Block_R3D)
    net.load_state_dict(torch.load(FILE))
    net.eval()
    net = net.to(device)


#Loading Dataloaders
train_loader, valid_loader, test_loader, clas_dataset =dataset.dataset3D(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor+20)

#######################################################
#                  Loading clasification image
#######################################################

##Classification Input Data
input_shape=torch.zeros(10,1,20,210,210)
clas_image_AD, clas_image_NC = dataset.clas_output3D(clas_dataset,input_shape)
clas_image_AD=clas_image_AD.to(device)
clas_image_NC=clas_image_NC.to(device)

print("Initialitaion finished", flush=True)

###################################################
#          Calculation of classification vector
###################################################

outputAD = net.forward_once(clas_image_AD)
outputNC = net.forward_once(clas_image_NC)
feature_AD=torch.mean(outputAD,dim=0)
feature_NC=torch.mean(outputNC,dim=0)

#######################################################
#                  Testing
#######################################################

with torch.no_grad():
    correct = 0
    total = 0


    for i, data in enumerate(test_loader, 0):
        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)

        output1= net.forward_once(inputs)
        
        euclidean_distance_AD = F.pairwise_distance(output1, feature_AD)    
        euclidean_distance_NC = F.pairwise_distance(output1, feature_NC)

        predicted_labels = torch.ge(euclidean_distance_AD,euclidean_distance_NC)*1
        
        correct_tensor=torch.eq(labels,predicted_labels)
    
        correct_run = torch.sum(correct_tensor)
        correct += correct_run
        total += torch.numel(labels)

    test_accuracy=correct/total
    print("")
    print("   ->Test Accuracy  :",test_accuracy.item(), flush=True)

    gc.collect()

print('=> ---- Finished Testing ---- ')

#######################################################
# Plotting feature vectors of classification reference
#######################################################

if plot_feature_vectors==1:

    plt.figure(0)

    for i in range(10):
        plt.plot(outputAD[i].cpu().detach().numpy(), label='{}'.format(i))
        #plt.plot(outputNC[i].cpu().detach().numpy(), label='NC')

    plt.plot(feature_AD.cpu().detach().numpy(), label='AD',color='black',linewidth='4')
    plt.legend(loc='lower right', bbox_to_anchor=(-0.1, 0))
    plt.savefig('PlotAD_predict.png',bbox_inches='tight')

    plt.figure(1)
    for i in range(10):
        plt.plot(outputNC[i].cpu().detach().numpy(), label='{}'.format(i))
        #plt.plot(outputNC[i].cpu().detach().numpy(), label='NC')


    plt.plot(feature_NC.cpu().detach().numpy(), label='NC',color='black',linewidth='4')
    plt.legend(loc='lower right', bbox_to_anchor=(-0.1, 0))
    plt.savefig('PlotNC_predict.png',bbox_inches='tight')

    plt.figure(2)
    plt.plot(feature_AD.cpu().detach().numpy(), label='AD',color='black',linewidth='4')
    plt.plot(feature_NC.cpu().detach().numpy(), label='NC',color='red',linewidth='1')
    plt.legend(loc='lower right', bbox_to_anchor=(-0.1, 0))
    plt.savefig('Plot_predict.png',bbox_inches='tight')
    print('=> ---- Finished Plotting feature vectors ---- ')    

