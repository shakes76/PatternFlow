from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib

import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import glob
from PIL import Image

import torchio as tio
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from model import UNet3D

'''
Class for calculating Dice Similarity Coefficient
'''
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    '''
    calculate dsc per label
    '''
    def single_loss(self, inputs, targets, smooth=0.1):
        intersection = (inputs * targets).sum()                            
        dice = (2.* intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return dice

    '''
    calculate dsc for each channel, add them up and get the mean
    '''
    def forward(self, inputs, targets, smooth=0.1):    
        
        input0 = (inputs.argmax(1) == 0)
        input1 = (inputs.argmax(1) == 1)
        input2 = (inputs.argmax(1) == 2)
        input3 = (inputs.argmax(1) == 3)
        input4 = (inputs.argmax(1) == 4)
        input5 = (inputs.argmax(1) == 5)

        target0 = (targets == 0)
        target1 = (targets == 1)
        target2 = (targets == 2)
        target3 = (targets == 3)
        target4 = (targets == 4)
        target5 = (targets == 5)
        
        dice0 = self.single_loss(input0, target0)
        dice1 = self.single_loss(input1, target1)
        dice2 = self.single_loss(input2, target2)
        dice3 = self.single_loss(input3, target3)
        dice4 = self.single_loss(input4, target4)
        dice5 = self.single_loss(input5, target5)
        
        dice = (dice0 + dice1 + dice2 + dice3 + dice4 + dice5) / 6.0    
        
        return 1 - dice


'''
Class for loading data from nii.gz files and prepocessing the data
'''
class NiiImageLoader(DataLoader) :
    def __init__(self, image_path, mask_path):
        self.inputs = []
        self.masks = []

        self.totensor = transforms.ToTensor()
        self.resize = tio.CropOrPad((128,256,256))
        self.filp0 = tio.RandomFlip(axes = 0, flip_probability = 1)
        self.filp1 = tio.RandomFlip(axes = 1, flip_probability = 1)
        self.filp2 = tio.RandomFlip(axes = 2, flip_probability = 1)

        for f in sorted(glob.iglob(image_path)):
            self.inputs.append(f)

        for f in sorted(glob.iglob(mask_path)):
            self.masks.append(f)

    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image_p = self.inputs[idx]
        mask_p = self.masks[idx]

        image = nib.load(image_p)
        image = np.asarray(image.dataobj)

        mask = nib.load(mask_p)
        mask = np.asarray(mask.dataobj)

        
        #Resize the images
        image = self.totensor(image)
        image = image.unsqueeze(0)
        image = self.resize(image)
        image = image.data

        mask = self.totensor(mask)
        mask = mask.unsqueeze(0)
        mask = self.resize(mask)
        mask = mask.data
        
        
        #data augmentation
        i = random.randint(0,4)
        if i == 0 :
            image = self.filp0(image)
            mask = self.filp0(mask)
        elif i == 1 :
            image = self.filp1(image)
            mask = self.filp1(mask)
        elif i == 2 :
            image = self.filp2(image)
            mask = self.filp2(mask)


        return image, mask


'''
The training function, will train the model save it as 'net_paras.pth'
Loss Function : CrossEntropyLoss
Optimizer : Adam
Epoch : 8
Batch_size : 1
'''

def main() :
    os.chdir(os.path.dirname(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device, flush = True)

    torch.cuda.empty_cache()
    model = UNet3D().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 1
    epoch = 8

    dataset = NiiImageLoader("v1/semantic_MRs_anon/*.nii.gz", 
                      "v1/semantic_labels_anon/*.nii.gz")

    trainset, valset, testset = torch.utils.data.random_split(dataset, [179, 16, 16])

    trainloader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=True)

    valloader = DataLoader(valset, batch_size=batch_size,
                        shuffle=True)
                        
    testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=True)
    
    for i in range(epoch) :
        model.train()
        for index, data in enumerate(trainloader, 0) :
            image, mask = data
            image = image.float().to(device)
            mask = mask.long().to(device)
            mask = mask.squeeze(1)
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step()

        '''
        run the model on the val set after each train loop
        '''
        model.eval()
        size = len(valloader.dataset)
        num_batches = len(valloader)
        val_loss = 0
        dice_all = 0
        with torch.no_grad():
            for X, y in valloader:
                X = X.float().to(device)
                y = y.long().to(device)
                y = y.squeeze(1)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                dice_all += (1 - dice_loss(pred, y))
        val_loss /= num_batches
        dice_all /= num_batches
        print(f"Avg loss: {val_loss:>8f}", flush = True)
        print(f"DSC: {dice_all:>8f} \n", flush = True)

        print('One Epoch Finished', flush = True)
        torch.save(model.state_dict(), 'net_paras.pth')

    '''
    run on test set after the train is finished
    '''
    model.eval()
    num_batches = len(testloader)
    dice_all = 0

    with torch.no_grad():
        for X, y in testloader:
            X = X.float().to(device)
            y = y.long().to(device)
            y = y.squeeze(1)
            pred = model(X)
            dice_all += (1 - dice_loss(pred, y))
    dice_all = dice_all / num_batches
    print(f"Dice: \n DSC: {dice_all:>8f} \n", flush = True)

if __name__ == "__main__":
    main()
