#train.py

#Containing the source code for training, validating, testing and saving your model.
#The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
#Make sure to plot the losses and metrics during training.



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
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model.Net()
net = net.to(device)

#Constants
epoch_range = 2
modulo=1 #Print frequency while training
batch_size=128

#Importing Custom Dataloader
import dataset_sim as data
train_loader, valid_loader, test_loader =data.dataset(batch_size,TRAIN_SIZE = 10, VALID_SIZE= 10, TEST_SIZE=10)




#Optimizer

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())


#Loss tracking during training
training_loss= torch.zeros(epoch_range)
testing_loss= torch.zeros(epoch_range)
test_accuracy=torch.zeros(epoch_range)


#xy,yz = data.mean_std_calculation(valid_loader)
#print(xy)
#print(yz)


image1,image2 = data.classification_data()
print(image2)
print(image1)




for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    print(f'EPOCH NUMBER: {epoch}', end ="", flush=True) 


    total = 0

    for i, data in enumerate(train_loader, 0):
        

        inputs_1= data[0].to(device) 
        inputs_2= data[1].to(device) 

        labels= data[2].to(device).to(torch.float32)

        
        #print(torch.sum(inputs_2-inputs_1))
        #print(labels)    
        #zero gradients
        optimizer.zero_grad()
        #print(torch.sum(torch.isnan(inputs_1)))
        #print(torch.max(inputs_2))
        # Optimization
        outputs = net(inputs_1,inputs_2).squeeze(1)
        
        if torch.sum(torch.isnan(outputs))>0:
            print(outputs)

       
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_value_(net.parameters(), 0.1)

        #optimizer.step()

        #Training loss
        training_loss[epoch]=training_loss[epoch]+loss#.detach().item()
        total += batch_size

        #Update where running
        if i % (modulo) == modulo-1:
            print("-", end ="", flush=True)
    
    print("")

    training_loss[epoch]=training_loss[epoch]/total
    print(f'Training Loss: {training_loss[epoch]}')

    del inputs_1, inputs_2, labels, outputs, loss, total
    gc.collect()
    




# No backpropagtion , No need for calculating gradients, => Faster calculation
with torch.no_grad():
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):

        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        outputs = net(inputs).squeeze(1)
        
        #outputs= torch.round(outputs)
        #correct += (outputs == labels).sum().item()

        total += batch_size

        #Testing loss to compare it to training loss
        loss = criterion(outputs, labels)
        testing_loss[epoch]=testing_loss[epoch]+loss.detach().item()

    test_accuracy[epoch] = (correct/total)
    testing_loss[epoch]=testing_loss[epoch]/total
    print(f'Testing Loss: {testing_loss[epoch]}')
    print(f'Test Accuracy: {test_accuracy[epoch]}', flush=True)


    #if epoch % 50 == 49:
    #    torch.save(net.state_dict(), 'checkpoint_9.pth')
          
        #   torch.save(training_loss, 'training_loss_18.pt')
        #   torch.save(test_accuracy, 'test_accuracy_18.pt') 
        #   torch.save(testing_loss, 'testing_loss_18.pt')

    del inputs_1, inputs_2, labels, outputs, loss, total, correct
    gc.collect()
    


print('Finished Training')

