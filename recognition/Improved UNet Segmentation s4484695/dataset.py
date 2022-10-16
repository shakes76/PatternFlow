import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
from torchvision.io import read_image

# Ensure to specify transforms for DataSet or otherwise will default to no transforms applied.
# Complete Paths should be provided for labels and images folder.
class ISIC2017DataSet(Dataset):
    def __init__(self, imgs_path, labels_path, transform=None):
        self.LabelsPath = labels_path
        self.ImagesPath = imgs_path
        # os.listdir does not gaurantee order.
        self.LabelNames = os.listdir(self.LabelsPath)
        self.imageNames = os.listdir(self.ImagesPath)
        self.LabelsSize = len(self.LabelNames)
        self.ImagesSize = len(self.imageNames)
        self.transform = transform

    def __len__(self):
        if self.ImagesSize != self.LabelsSize:
            print("Bad Data! Please Check Data, or unpredictable behaviour!")
            return -1
        else:
            return self.ImagesSize

    # Using os.path.join with self.imageNames + "_segmentation" ensures img_path and label_path both refer to same sample. 
    # This is required as os.listdir does not gaurantee order.
    def __getitem__(self, idx):
        img_path = os.path.join(self.ImagesPath, self.imageNames[idx])
        image = read_image(img_path)
        label_path = os.path.join(self.LabelsPath, self.imageNames[idx] + "_segmentation")
        label = read_image(label_path)

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def ISIC_Transform_Train(self):
        
        transformTrain = transforms.Compose([
            transforms.ToTensor(),
            
        ])

        return transformTrain

    def ISIC_Transform_Test(self):
        
        transformTest = transforms.Compose([
            transforms.ToTensor(),
            
        ])

        return transformTest

    def ISIC_Transform_Valid(self):
        
        transformValid = transforms.Compose([
            transforms.ToTensor(),
            
        ])

        return transformValid