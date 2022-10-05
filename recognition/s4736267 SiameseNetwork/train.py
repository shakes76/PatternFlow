#train.py

#Containing the source code for training, validating, testing and saving your model.
#The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
#Make sure to plot the losses and metrics during training.


#Defining CNN

import torch
from sklearn.datasets import fetch_lfw_people

#Defining CNN

import torch.nn as nn
import torch.nn.functional as F

#Solver

import torch.optim as optim

#
import torchvision
import torchvision.transforms as transforms

import math

import gc 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Data Set CIFAR10 
#transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
#,transforms.RandomRotation((-60,60))
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#transform_train = transforms.Compose([transforms.RandomApply(([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)]), p=0.5),transforms.RandomGrayscale(p=0.1),transforms.RandomRotation((-60,60)),transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(*stats,inplace=True)])
transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(*stats)])
batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=3)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=3)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class Residual_Identity_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Residual_Identity_Block, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm2d(c_in),
                            nn.ReLU())
        self.branch     = nn.Sequential(
                            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1), 
                            nn.BatchNorm2d(c_out),
                            nn.ReLU(),
                            nn.Conv2d(c_out, c_out,kernel_size=3, stride=1, padding=1))       
    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+x

        return x

class Residual_Conv_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Residual_Conv_Block, self).__init__()
        self.block_prep = nn.Sequential(
                            nn.BatchNorm2d(c_in),
                            nn.ReLU())
        self.branch = nn.Sequential(
                            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), 
                            nn.BatchNorm2d(c_out),
                            nn.ReLU(),
                            nn.Conv2d(c_out, c_out,kernel_size=3, stride=1, padding=1))
        self.conv       = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = self.block_prep(x)
        x = self.branch(x)+self.conv(x)

        return x

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes, identity_block, conv_block):
        super().__init__()
        
        self.prep = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), 
                                  nn.BatchNorm2d(64), 
                                  nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))

        #As stated in the torch.nn.CrossEntropyLoss() doc:

        #This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

        #Therefore, you should not use softmax before.
        #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

        

        self.block0_1 = self._make_residual_block_(identity_block, 64, 64)
        self.block1_1 = self._make_residual_block_(identity_block, 64, 64)

        self.block0_2 = self._make_residual_block_(conv_block, 64, 128)
        self.block1_2 = self._make_residual_block_(identity_block, 128, 128)

        self.block0_3 = self._make_residual_block_(conv_block, 128, 256)
        self.block1_3 = self._make_residual_block_(identity_block, 256, 256)

        self.block0_4 = self._make_residual_block_(conv_block, 256, 512)
        self.block1_4 = self._make_residual_block_(identity_block, 512, 512)



    def _make_residual_block_(self, block, c_in, c_out):
        layers = []
        layers.append(block(c_in,c_out))

        return nn.Sequential(*layers)  



    def forward(self, x):

        out = self.prep(x)

#layer1
        out = self.block0_1(out) 
        out = self.block1_1(out) 
#layer2
        out = self.block0_2(out) 
        out = self.block1_2(out)    
#layer1
        out = self.block0_3(out) 
        out = self.block1_3(out) 
#layer2
        out = self.block0_4(out) 
        out = self.block1_4(out)


        out = self.classifier(out)
        return out



net = ResNet18(3, 10,Residual_Identity_Block,Residual_Conv_Block)


net = net.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=0.0001)

print(">>> Set-up finished <<<", flush=True)

epoch_range=100

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_range, eta_min=0.0005)

modulo=torch.tensor(round(50048/(batch_size*10))).to(device)
training_loss= torch.zeros(epoch_range).to(device)
testing_loss= torch.zeros(epoch_range).to(device)
test_accuracy=torch.zeros(epoch_range).to(device)


for epoch in range(epoch_range):  # loop over the dataset multiple times
    
    print(f'EPOCH NUMBER: {epoch}', end ="", flush=True)    
    training_loss[epoch] = 0.0
    total = torch.tensor(0).to(device)
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        #inputs, labels = data[0], data[1]

        inputs= data[0].to(device)   #torch.cat((inputs, inputs.permute(0,1,3,2), torch.flip(inputs, dims=(3,)), torch.flip(inputs, dims=(2,))), 0).to(device)
        labels= data[1].to(device)   #torch.cat((labels, labels, labels, labels), 0).to(device)
        # zero gradients
        optimizer.zero_grad()

        # Optimization
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        #nn.utils.clip_grad_value_(net.parameters(), 0.1)


        optimizer.step()

        #Training loss
        training_loss[epoch]=training_loss[epoch]+loss.detach().item()
        total += batch_size

        #Update where running
        if i % (modulo) == modulo-1:
            print("-", end ="", flush=True)
    #print(total)
    print("")
    training_loss[epoch]=training_loss[epoch]/total
    print(f'Training Loss: {training_loss[epoch]}')

    del inputs, labels, outputs, loss, total

    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)

    # No backpropagtion , No need for calculating gradients, => Faster calculation
    with torch.no_grad():
      for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        # _, not uses (max value)
        # predicted stores the indice => equals class

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        total += batch_size

        #Testing loss to compare it to training loss
        loss = criterion(outputs, labels)
        testing_loss[epoch]=testing_loss[epoch]+loss.detach().item()

    test_accuracy[epoch] = (correct/total)
    testing_loss[epoch]=testing_loss[epoch]/total
    print(f'Testing Loss: {testing_loss[epoch]}')
    print(f'Test Accuracy: {test_accuracy[epoch]}', flush=True)


    if epoch % 50 == 49:
        torch.save(net.state_dict(), 'checkpoint_9.pth')
        
    torch.save(training_loss, 'training_loss_18.pt')
    torch.save(test_accuracy, 'test_accuracy_18.pt') 
    torch.save(testing_loss, 'testing_loss_18.pt')

        
    del inputs, labels, outputs, loss, predicted, correct, total
    gc.collect()
    scheduler.step() 

print('Finished Training')

torch.save(net.state_dict(), 'checkpoint_18.pth')
torch.save(training_loss, 'training_loss_18.pt')
torch.save(test_accuracy, 'test_accuracy_18.pt') 
torch.save(testing_loss, 'testing_loss_18.pt')

print('Finished Downloading')