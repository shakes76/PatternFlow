
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

net = model.Net()
net = net.to(device)

#Constants
epoch_range = 20

batch_size=128
train_factor=1#00
test_factor=1#0

modulo=round(train_factor/10) +1#Print frequency while training

#Importing Custom Dataloader
import dataset as data
train_loader, valid_loader, test_loader =data.dataset(batch_size,TRAIN_SIZE = batch_size*train_factor, VALID_SIZE= 100, TEST_SIZE=batch_size*test_factor)



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive



#Optimizer
criterion = ContrastiveLoss()
#criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)


#Loss tracking during training
training_loss= torch.zeros(epoch_range)
testing_loss= torch.zeros(epoch_range)
test_accuracy=torch.zeros(epoch_range)


#xy,yz = data.mean_std_calculation(valid_loader)
#print(xy, flush=True)
#print(yz, flush=True)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)
        
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        nn.init.normal_(m.bias.data, mean=0.5, std=0.01)
    
net.apply(init_weights)

class_image_NC , class_image_AD = data.classification_data()

class_image_NC = class_image_NC[None, :].to(device) 
class_image_AD = class_image_AD[None, :].to(device) 

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

        #nn.utils.clip_grad_value_(net.parameters(), 100000)

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
        #outputs = net.forward_once(inputs).squeeze(1)
        output1,output2 = net(inputs,class_image_NC)#.squeeze(1)
        euclidean_distance_NC = F.pairwise_distance(output1, output2)

        

        output1,output2 = net(inputs,class_image_AD)#.squeeze(1)
        euclidean_distance_AD = F.pairwise_distance(output1, output2)
        euclidean_distance_MAX = torch.ge(euclidean_distance_AD,euclidean_distance_NC)
        
        print("MAX",euclidean_distance_MAX)

        #print(euclidean_distance_NC)
        #print(euclidean_distance_AD)
        print("labels",labels)

        correct_tensor=torch.eq(labels,euclidean_distance_MAX)

        print("correct tensor", correct_tensor)

        correct_run = torch.sum(correct_tensor)
        print("correct_run", correct_run)
        #outputs= torch.round(outputs)
        correct += correct_run
        print(correct)

        total += batch_size

        #Testing loss to compare it to training loss
        
        #loss = criterion(outputs, labels)
        #testing_loss[epoch]=testing_loss[epoch]+loss.detach().item()

    test_accuracy[epoch] = (correct/total)
    testing_loss[epoch]=testing_loss[epoch]/total
    print(f'Testing Loss: {testing_loss[epoch]}')
    print(f'Test Accuracy: {test_accuracy[epoch]}', flush=True)


    #if epoch % 50 == 49:
    #    torch.save(net.state_dict(), 'checkpoint_9.pth')
          
        #   torch.save(training_loss, 'training_loss_18.pt')
        #   torch.save(test_accuracy, 'test_accuracy_18.pt') 
        #   torch.save(testing_loss, 'testing_loss_18.pt')

    del inputs, labels, output1, output2, total, correct
    gc.collect()
    


print('Finished Training')

