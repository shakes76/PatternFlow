
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = modules.Net_3D()
net.load_state_dict(torch.load('sim_net_3D.pt')) #change to .pt
net.eval()
net = net.to(device)

net_clas = modules.Net_clas3D()
net_clas = net_clas.to(device)

#Constants
epoch_range = 3#30#00


batch_size=20*4
train_factor=1000#00
test_factor=400#0
valid_factor=10

modulo=round(train_factor*20/(batch_size*10))+1 #Print frequency while training

#Importing Custom Dataloader
import dataset 
train_loader, valid_loader, test_loader, clas_dataset =dataset.dataset3D(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor+20)


#Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(net_clas.parameters(),lr = 0.001)
scheduler = StepLR(optimizer, step_size=batch_size, gamma=0.95)

#Loss tracking during training
training_loss= torch.zeros(epoch_range)
testing_loss= torch.zeros(epoch_range)
test_accuracy=torch.zeros(epoch_range)

input_shape=torch.zeros(1,20,210,210)

clas_image_AD, clas_image_NC = dataset.clas_output3D(clas_dataset,input_shape)
clas_image_AD=clas_image_AD.to(device)
clas_image_NC=clas_image_NC.to(device)

print("Initialitaion finished", flush=True)

for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    print(f'EPOCH NUMBER: {epoch} =', end ="", flush=True) 


    total = 0

    for i, data in enumerate(train_loader, 0):
        

        inputs= data[0].to(device) 
        labels= data[3].to(device).to(torch.float32)
                    
        feauture_image_AD = net.forward_once(clas_image_AD)
        feauture_image_NC = net.forward_once(clas_image_NC)
        feauture_inputs   = net.forward_once(inputs)

        #zero gradients
        optimizer.zero_grad()

        output = net_clas(feauture_inputs,feauture_image_AD,feauture_image_NC).squeeze(1)

        
        loss = criterion(output,labels)
        loss.backward()

        #nn.utils.clip_grad_value_(net.parameters(), 0.1)

        optimizer.step()

        #Training loss
        training_loss[epoch]=training_loss[epoch]+loss#.detach().item()
        total += torch.numel(labels)

        #scheduler.step()

        #Update where running
        if i % (modulo) == modulo-1:
            print("-", end ="", flush=True)
    
        print("")
        print("output ",output[0:20], "labels",labels[0:20])

    training_loss[epoch]=training_loss[epoch]/total
    print(f'Training Loss: {training_loss[epoch]}')

    del inputs, labels, output, loss, total
    gc.collect()
    
####


# No backpropagtion , No need for calculating gradients, => Faster calculation
with torch.no_grad():
    correct = 0
    total = 0


    for i, data in enumerate(test_loader, 0):

        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        
        #print("labels",labels)
        #print("slice_number train.py",slice_number)

        mode=0
        if mode == 0:

            print("labels",labels)
            print("CNN output",net_clas(inputs))
            print("CNN output",torch.round(net_clas(inputs)))

            predicted_labels = torch.round(net_clas(inputs))
            correct_tensor=torch.eq(labels,predicted_labels)
    
            correct_run = torch.sum(correct_tensor)
            correct += correct_run
            total += torch.numel(labels)

        else:
            predicted_labels = torch.ge(euclidean_distance_AD,euclidean_distance_NC)*1
            number_sets=round(torch.numel(labels)/20)

            predicted_labels_patient=torch.zeros(number_sets)
            labels_patient=torch.zeros(number_sets)

            for j in range(number_sets):
                
                predicted_labels_patient[j]= torch.round(torch.sum(predicted_labels[j:j+20])/20)
                labels_patient[j] = labels[j*20]
                print("predicted_labels_patient", predicted_labels_patient[j],"--",torch.sum(predicted_labels[j:j+20])/20,"   labels patient",labels_patient[j])
                
            correct_tensor=torch.eq(labels_patient,predicted_labels_patient)
            correct_run = torch.sum(correct_tensor)

            correct += correct_run
            total += number_sets

            print("correct_run",correct_run)
            print("")


    test_accuracy = (correct/total)
    print(f'Test Accuracy: {test_accuracy}', flush=True)


    #if epoch % 50 == 49:
    #    torch.save(net.state_dict(), 'checkpoint_9.pth')
          
        #   torch.save(training_loss, 'training_loss_18.pt')
        #   torch.save(test_accuracy, 'test_accuracy_18.pt') 
        #   torch.save(testing_loss, 'testing_loss_18.pt')

    #del inputs, labels,predicted_labels, output1, output2, total, correct, euclidean_distance_AD, euclidean_distance_NC
    gc.collect()
    


print('Finished Training')

