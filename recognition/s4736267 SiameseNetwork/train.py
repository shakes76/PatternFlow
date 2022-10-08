
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

net = modules.Net_batchnom()
net = net.to(device)

#Constants
epoch_range = 5#30#00


batch_size=20
train_factor=1000#00
test_factor=400#0
valid_factor=10

modulo=round(train_factor*20/(batch_size*10))+1 #Print frequency while training

#Importing Custom Dataloader
import dataset 
train_loader, valid_loader, test_loader, clas_dataset =dataset.dataset(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor+20)





#Optimizer
criterion = modules.ContrastiveLoss()
#criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(),lr = 0.01)
scheduler = StepLR(optimizer, step_size=batch_size, gamma=0.7)


#Loss tracking during training
training_loss= torch.zeros(epoch_range)
testing_loss= torch.zeros(epoch_range)
test_accuracy=torch.zeros(epoch_range)


#xy,yz = dataset.mean_std_calculation(train_loader)
#dataset.crop_area_pos(train_loader)   not working !!

#net.apply(modules.init_weights)




print("Initialitaion finished", flush=True)

for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    print(f'EPOCH NUMBER: {epoch} =', end ="", flush=True) 


    total = 0

    for i, data in enumerate(train_loader, 0):
        

        inputs_1= data[0].to(device) 
        inputs_2= data[1].to(device) 

        labels= data[2].to(device).to(torch.float32)

        
        #zero gradients
        optimizer.zero_grad()

        output1,output2 = net(inputs_1,inputs_2)#.squeeze(1)

        loss = criterion(output1,output2,labels)
        loss.backward()

        nn.utils.clip_grad_value_(net.parameters(), 0.1)

        optimizer.step()

        #Training loss
        training_loss[epoch]=training_loss[epoch]+loss#.detach().item()
        total += torch.numel(labels)

        scheduler.step()

        #Update where running
        if i % (modulo) == modulo-1:
            print("-", end ="", flush=True)
    
    print("")

    training_loss[epoch]=training_loss[epoch]/total
    print(f'Training Loss: {training_loss[epoch]}')

    del inputs_1, inputs_2, labels, output1, output2, loss, total
    gc.collect()
    
####


# No backpropagtion , No need for calculating gradients, => Faster calculation
with torch.no_grad():
    correct = 0
    total = 0


    for i, data in enumerate(test_loader, 0):

        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        slice_number = data[2].to(device) 
        #print("labels",labels)
        #print("slice_number train.py",slice_number)

        clas_image_AD, clas_image_NC = dataset.clas_output(clas_dataset,slice_number,inputs)

        output1,output2 = net(inputs,clas_image_AD)#.squeeze(1)
        print(torch.sum(output1-output2))
        euclidean_distance_AD = F.pairwise_distance(output1, output2)    

        output1,output2 = net(inputs,clas_image_NC)#.squeeze(1)
        euclidean_distance_NC = F.pairwise_distance(output1, output2)


        mode=0
        if mode == 0:

            predicted_labels = torch.ge(euclidean_distance_AD,euclidean_distance_NC)*1
        

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

        #print("euc_NC",euclidean_distance_NC)
        #print("euc_AD",euclidean_distance_AD)
        #print("lab",labels)
        #print("pred_lab", predicted_labels)
        #print("cor_ten", correct_tensor)    

    test_accuracy = (correct/total)
    print(f'Test Accuracy: {test_accuracy}', flush=True)


    #if epoch % 50 == 49:
    #    torch.save(net.state_dict(), 'checkpoint_9.pth')
          
        #   torch.save(training_loss, 'training_loss_18.pt')
        #   torch.save(test_accuracy, 'test_accuracy_18.pt') 
        #   torch.save(testing_loss, 'testing_loss_18.pt')

    del inputs, labels,predicted_labels, output1, output2, total, correct, euclidean_distance_AD, euclidean_distance_NC
    gc.collect()
    


print('Finished Training')

