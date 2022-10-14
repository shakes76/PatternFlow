
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
gc.collect()
torch.cuda.empty_cache()
#Importing CNN Model
import modules

#save image
from torchvision.utils import save_image

#plot
import matplotlib.pyplot as plt
import numpy as np

#Scheduler
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#net = modules.Net_batchnom()
#net = modules.Net_3D()
#net = modules.ResNet18_3D(modules.Residual_Identity_Block,modules.Residual_Conv_Block)
net = modules.ResNet18_R3D(modules.Residual_Identity_Block_R3D,modules.Residual_Conv_Block_R3D)
#net.load_state_dict(torch.load('sim_net_3D_R3D_hab8.pt')) #change to .pt
#net.eval()
net = net.to(device)


#torch.save(net.state_dict(), 'sim_net_ResNet.pt')

#Constants
epoch_range = 20


batch_size=8
train_factor=1000
test_factor=400
valid_factor=10

modulo=round(train_factor*20/(batch_size*10))+1 #Print frequency while training

#Importing Custom Dataloader
import dataset 
train_loader, valid_loader, test_loader, clas_dataset =dataset.dataset3D(batch_size,TRAIN_SIZE = 20*train_factor, VALID_SIZE= 20*valid_factor, TEST_SIZE=20*test_factor+20)


#Optimizer
criterion = modules.ContrastiveLoss()
#criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)


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


        output1,output2 = net(inputs_1,inputs_2)

        loss = criterion(output1,output2,labels)
        loss.backward()

        #nn.utils.clip_grad_value_(net.parameters(), 0.1)

        optimizer.step()

        #Training loss
        training_loss[epoch]=training_loss[epoch]+loss#.detach().item()
        total += torch.numel(labels)

        scheduler.step()

        #print("LOSS:",loss.item(), flush=True)
        #Update where running
        #if i % (modulo) == modulo-1:
        #print("-", end ="", flush=True)
        #print("sim:",torch.sum(F.pairwise_distance(output1[0], output2[0],p=1.0)).item()/512," with label:",labels[0].item() ,"   loss:",criterion(output1[0],output2[0],labels[0]).item(), flush=True)
        
    print("")


    training_loss[epoch]=training_loss[epoch]/total
    print(f'Training Loss: {training_loss[epoch]}')
    



    del inputs_1, inputs_2, labels, output1, output2, loss, total
    gc.collect()
    
####


input_shape=torch.zeros(10,1,20,210,210)

clas_image_AD, clas_image_NC = dataset.clas_output3D(clas_dataset,input_shape)

clas_image_AD=clas_image_AD.to(device)
clas_image_NC=clas_image_NC.to(device)
# No backpropagtion , No need for calculating gradients, => Faster calculation

outputAD,outputNC = net(clas_image_AD,clas_image_NC)

feature_AD=torch.sum(outputAD, dim=0)/10
feature_NC=torch.sum(outputNC, dim=0)/10

if 1==1:
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
    plt.savefig('Plot.png',bbox_inches='tight')




with torch.no_grad():
    correct = 0
    total = 0


    for i, data in enumerate(test_loader, 0):
        inputs= data[0].to(device) 
        
        labels= data[1].to(device).to(torch.float32)
        #slice_number = data[2].to(device) 
        #print("labels",labels)
        #print("slice_number train.py",slice_number)

        


        output1= net.forward_once(inputs)#.squeeze(1)
        
        #print(output1.shape)
        #print(torch.sum(output1-output2))
        euclidean_distance_AD = F.pairwise_distance(output1, feature_AD)    

        #print(euclidean_distance_AD.shape)
        output1,output2 = net(inputs,clas_image_NC)#.squeeze(1)
        euclidean_distance_NC = F.pairwise_distance(output1, feature_NC)


        predicted_labels = torch.ge(euclidean_distance_AD,euclidean_distance_NC)*1
        

        correct_tensor=torch.eq(labels,predicted_labels)
    

        correct_run = torch.sum(correct_tensor)
        correct += correct_run
        total += torch.numel(labels)



        print("euc_NC",euclidean_distance_NC)
        print("euc_AD",euclidean_distance_AD)
        print("lab",labels)
        print("pred_lab", predicted_labels)
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
    

torch.save(net.state_dict(), 'sim_net_3D_loss.pt')
print('Finished Training')

