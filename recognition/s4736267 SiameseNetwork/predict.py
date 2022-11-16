
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

epoch_range = 300
epoch_range_clas = 30

batch_size=8
train_factor=10000               #Number of Persons
test_factor=400
valid_factor=400

FILE_SNN_LOAD   ="SNN_weights.pth"                   #Load location of pre-trained ResNet 
FILE_CLAS_LOAD  ="CLAS_weights.pth"                   #Load location of pre-trained Classification Net 


#(0=disabled)
plot_feature_vectors=0          #Ploting feature vectors



#######################################################
# Loading Net and defining optimizer, scheduler and weight initialisation
#######################################################

#Garbage collection
gc.collect()
torch.cuda.empty_cache()

#Importing CNN Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = modules.ResNet_3D(modules.Residual_Identity_Block_R3D,modules.Residual_Conv_Block_R3D)
net.load_state_dict(torch.load(FILE_SNN_LOAD))
net = net.to(device)
net.eval()

net_clas = modules.Net_clas3D()
net_clas.load_state_dict(torch.load(FILE_CLAS_LOAD))
net_clas = net.to(device)
net_clas.eval()


#Loading Dataloaders
train_loader, valid_loader, test_loader, train_loader_clas, valid_loader_clas, dataset_FV =dataset.dataset3D(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor)

#Mean and deviation calculation
#xy,yz = dataset.mean_std_calculation(train_loader)
#dataset.crop_area_pos(train_loader)   not working !!

print("Initialitaion finished", flush=True)
print("")

#######################################################
#                  Testing
#######################################################

print("Start testing", flush=True)
with torch.no_grad():
    correct = 0
    total = 0
    net_clas.eval()

    for i, data in enumerate(test_loader, 0):
        
        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        
        feauture_inputs   = net.forward_once(inputs)
        output = net_clas(feauture_inputs)#.squeeze(0)
        output=output.squeeze(1)

        predicted_labels = torch.round(output)
        correct_tensor=torch.eq(labels,predicted_labels)
    
        correct_run = torch.sum(correct_tensor)
        correct += correct_run
        total += torch.numel(labels)

    test_accuracy=correct/total
    print("->Test Accuracy  :",test_accuracy.item(), flush=True)

    gc.collect()

print('=> ---- Finished Testing ---- ', flush=True)
print("")

#######################################################
# Plotting feature vectors of classification reference
#######################################################

if plot_feature_vectors==1:
    print("Start plotting feature vectors", flush=True)
    
    ##Classification Input Data
    input_shape=torch.zeros(10,1,20,210,210)
    FV_image_AD, FV_image_NC = dataset.clas_outputFV(dataset_FV,input_shape)
    FV_image_AD=FV_image_AD.to(device)
    FV_image_NC=FV_image_NC.to(device)

    outputAD = net.forward_once(FV_image_AD)
    outputNC = net.forward_once(FV_image_NC)
    feature_AD=torch.mean(outputAD,dim=0)
    feature_NC=torch.mean(outputNC,dim=0)

    plt.figure(0)

    for i in range(10):
        plt.plot(outputAD[i].cpu().detach().numpy(), label='{}'.format(i))
        #plt.plot(outputNC[i].cpu().detach().numpy(), label='NC')

    plt.plot(feature_AD.cpu().detach().numpy(), label='AD',color='black',linewidth='4')
    plt.legend(loc='lower right', bbox_to_anchor=(-0.1, 0))
    plt.savefig('PlotAD.png',bbox_inches='tight')

    plt.figure(1)
    for i in range(10):
        plt.plot(outputNC[i].cpu().detach().numpy(), label='{}'.format(i))
        #plt.plot(outputNC[i].cpu().detach().numpy(), label='NC')


    plt.plot(feature_NC.cpu().detach().numpy(), label='NC',color='black',linewidth='4')
    plt.legend(loc='lower right', bbox_to_anchor=(-0.1, 0))
    plt.savefig('PlotNC.png',bbox_inches='tight')

    plt.figure(2)
    plt.plot(feature_AD.cpu().detach().numpy(), label='AD',color='black',linewidth='4')
    plt.plot(feature_NC.cpu().detach().numpy(), label='NC',color='red',linewidth='1')
    plt.legend(loc='lower right', bbox_to_anchor=(-0.1, 0))
    plt.savefig('PlotADNC.png',bbox_inches='tight')
    print('=> ---- Finished Plotting feature vectors ---- ')    

