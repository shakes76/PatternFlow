
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
FILE_SNN_UPDATE ="SNN_weights_improved.pth"     #Save location of trained ResNet => only if training is continued

FILE_CLAS_LOAD  ="CLAS_weights.pth"                   #Load location of pre-trained Classification Net 
FILE_CLAS_UPDATE="CLAS_weights_improved.pth"     #Save location of trained Classification Net => only if training is continued

#(0=disabled)
load_pretrained_model=1
training=0

load_pretrained_model_clas=0
training_clas=1

scheduler_active=1
gradient_clipping=0             #Gradient clippling    

scheduler_active_clas=1
gradient_clipping_clas=0        #Gradient clippling  

plot_feature_vectors=0          #Ploting feature vectors
plot_loss = 0                   #Ploting training and validation loss


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
    #Weight initialisation
    #net.apply(modules.init_weights)
else:
    net = modules.ResNet_3D(modules.Residual_Identity_Block_R3D,modules.Residual_Conv_Block_R3D)
    net.load_state_dict(torch.load(FILE_SNN_LOAD))
    net = net.to(device)
    net.eval()

if load_pretrained_model_clas==0: 
    net_clas = modules.Net_clas3D()
    net_clas = net_clas.to(device)
else:
    net_clas = modules.Net_clas3D()
    net_clas.load_state_dict(torch.load(FILE_CLAS_LOAD))
    net_clas = net.to(device)
    net_clas.eval()


#Optimizer
criterion = modules.TripletLoss()
#criterion = torch.nn.TripletMarginLoss(margin=64.0)
optimizer = optim.Adam(net.parameters(),lr = 0.0001)

#Learning rate scheduler
scheduler = StepLR(optimizer, step_size=300, gamma=0.95)


#Optimizer Classifier 
criterion_clas = nn.BCELoss()
optimizer_clas = optim.Adam(net_clas.parameters(),lr = 0.005)

#Learning rate scheduler Classifier
scheduler_clas = StepLR(optimizer_clas, step_size=100, gamma=0.95)

#Loading Dataloaders
train_loader, valid_loader, test_loader, train_loader_clas, valid_loader_clas, dataset_FV =dataset.dataset3D(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor)

#Mean and deviation calculation
#xy,yz = dataset.mean_std_calculation(train_loader)
#dataset.crop_area_pos(train_loader)   not working !!

print("Initialitaion finished", flush=True)
print("")

#######################################################
#                  Training and validation of SNN
#######################################################

print("Start training SNN", flush=True)

#Loss tracking during training
training_loss= torch.zeros(epoch_range)
validation_loss= torch.zeros(epoch_range)
validation_accuracy=torch.zeros(epoch_range)

if training==0:
    epoch_range=0
    print("->Skip training SNN", flush=True)


for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    print(f'->EPOCH NUMBER: {epoch} =', end ="", flush=True) 

    #TRAINING
    net.train()
    total = 0
    loss_run = 0

    for i, data in enumerate(train_loader, 0):
        #Data Transfer
        inputs_1= data[0].to(device) 
        inputs_2= data[1].to(device) #positive
        inputs_3= data[2].to(device) #negative 
        
        #Zero gradients
        optimizer.zero_grad()

        #Loss calculation
        output_1,output_2,output_3 = net(inputs_1,inputs_2,inputs_3)

        loss = criterion(output_1,output_2,output_3)
        loss.backward()

        #Gradient Clipping
        if gradient_clipping>0:
            nn.utils.clip_grad_value_(net.parameters(), gradient_clipping)

        #Optimisation Step
        optimizer.step()

        #Training loss
        loss_run +=loss.detach().item()
        total += inputs_1.size(dim=0)

        #Learning rate scheduler
        if scheduler_active==1:
            scheduler.step()

        print("-",end ="",flush=True)

    training_loss[epoch]=loss_run/total
    print("")
    print("  ->Training Loss:",training_loss[epoch].item(), flush=True)
    gc.collect()
    
    #Validation
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        
        for i, data in enumerate(valid_loader, 0):

            #Data Transfer
            inputs_1= data[0].to(device) 
            inputs_2= data[1].to(device) #positive
            inputs_3= data[2].to(device) #negative 
        
        
            #Zero gradients
            optimizer.zero_grad()

            #Loss calculation
            output_1,output_2,output_3 = net(inputs_1,inputs_2,inputs_3)

            loss = criterion(output_1,output_2,output_3)

    
            #Training loss
            loss_run +=loss.detach().item()
            total += inputs_1.size(dim=0)

    validation_loss[epoch]=loss_run/total
    print("  ->Validation Loss:",validation_loss[epoch].item(), flush=True)

    # Plotting training and validation loss 

    if (plot_loss==1 and epoch%10==0) or epoch==(epoch_range-1):
        plt.figure(3)
        plt.plot(training_loss[0:epoch].detach().numpy(), label='Training_loss')
        plt.plot(validation_loss[0:epoch].detach().numpy(), label='Validation_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower center', bbox_to_anchor=(0.3, -0.3),ncol=2)
        plt.twinx()

        plt.savefig('Plot_Loss_SNN.png',bbox_inches='tight')

        print('  ->Finished Plotting loss  ', flush=True) 
        plt.clf()
    

    gc.collect()

###################################################
#                  Saving SNN parameters
###################################################

torch.save(training_loss, "training_loss_SNN.pth")
torch.save(validation_loss, "validation_loss_SNN.pth")

if training==1:
    torch.save(net.state_dict(), FILE_SNN_UPDATE)

print('=> ---- Finished Training SNN---- ', flush=True)

print("")

#######################################################
#                  Training and validation of Classifier Net
#######################################################

print("Start training Classifier", flush=True)

#Loss tracking during training
training_loss_clas= torch.zeros(epoch_range_clas)
validation_loss_clas= torch.zeros(epoch_range_clas)
validation_accuracy_clas=torch.zeros(epoch_range_clas)

if training_clas==0:
    epoch_range_clas=0
    print("->Skip training Classifier", flush=True)

for epoch in range(epoch_range_clas):  # loop over the dataset multiple times
    
    print(f'->EPOCH NUMBER: {epoch} =', end ="", flush=True) 

    #TRAINING
    net_clas.train()
    net.eval()
    total = 0
    loss_run = 0

    for i, data in enumerate(train_loader_clas, 0):
        
        #Data Transfer
        inputs= data[0].to(device) 
        labels= data[1].to(device).to(torch.float32)
                    
        feauture_inputs   = net.forward_once(inputs)

        #Zero gradients
        optimizer_clas.zero_grad()

        output = net_clas(feauture_inputs).squeeze(1)

    
        #Zero gradients
        optimizer_clas.zero_grad()

        #Loss calculation
        loss = criterion_clas(output,labels)
        loss.backward()

        #Gradient Clipping
        if gradient_clipping_clas>0:
            nn.utils.clip_grad_value_(net_clas.parameters(), gradient_clipping)

        #Optimisation Step
        optimizer_clas.step()

        #Training loss
        loss_run +=loss.detach().item()
        total += inputs.size(dim=0)

        #Learning rate scheduler
        if scheduler_active_clas==1:
            scheduler_clas.step()

        print("-",end ="",flush=True)

    training_loss_clas[epoch]=loss_run/total
    print("")
    print("  ->Training Classifier Loss  :",training_loss_clas[epoch].item(), flush=True)
    gc.collect()
    
    #Validation
    net_clas.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        
        for i, data in enumerate(valid_loader_clas, 0):

            #Data Transfer
            inputs= data[0].to(device) 
            labels= data[1].to(device).to(torch.float32)
        
            feauture_inputs   = net.forward_once(inputs)

            #Zero gradients
            optimizer_clas.zero_grad()

            #Loss calculation

            output = net_clas(feauture_inputs).squeeze(1)
            loss = criterion_clas(output,labels)

    
            #Training loss
            loss_run +=loss.detach().item()
            total += inputs.size(dim=0)

    validation_loss_clas[epoch]=loss_run/total
    print("  ->Validation Loss:",validation_loss_clas[epoch].item(), flush=True)

    # Plotting training and validation loss 

    if (plot_loss==1 and epoch%10==0) or epoch==(epoch_range-1):
        plt.figure(3)
        plt.plot(training_loss_clas[0:epoch].detach().numpy(), label='Training_loss')
        plt.plot(validation_loss_clas[0:epoch].detach().numpy(), label='Validation_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower center', bbox_to_anchor=(0.3, -0.3),ncol=2)
        plt.twinx()

        plt.savefig('Plot_Loss_Clas.png',bbox_inches='tight')

        print('  ->Finished Plotting loss ---- ', flush=True) 
        plt.clf()
    

    gc.collect()



###################################################
#                  Saving Classifier parameters
###################################################


torch.save(training_loss, "training_loss_CLAS.pth")
torch.save(validation_loss, "validation_loss_CLAS.pth")

if training_clas==1:
    torch.save(net.state_dict(), FILE_CLAS_UPDATE)

print('=> ---- Finished Training Classifier---- ', flush=True)


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

