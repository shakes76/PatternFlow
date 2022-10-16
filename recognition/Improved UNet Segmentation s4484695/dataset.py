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
    def __init__(self, imgs_path, labels_path, transform=None, labelTransform=None):
        self.LabelsPath = labels_path
        self.ImagesPath = imgs_path
        # os.listdir does not gaurantee order.
        self.LabelNames = os.listdir(self.LabelsPath)
        self.imageNames = os.listdir(self.ImagesPath)
        self.LabelsSize = len(self.LabelNames)
        self.ImagesSize = len(self.imageNames)
        self.transform = transform
        self.labelTransform = labelTransform

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
        label_path = os.path.join(self.LabelsPath, self.imageNames[idx].removesuffix(".jpg") + "_segmentation.png")
        label = read_image(label_path)

        if self.transform:
            image = self.transform(image)
        if self.labelTransform:
            label = self.labelTransform(label)
        
        return image, label
    
def ISIC_transform_img():
        
    transformTrain = transforms.Compose([
         transforms.Resize((512, 512)),
    ])

    return transformTrain

def ISIC_transform_label():
        
    transformTest = transforms.Compose([
        transforms.Resize((512, 512)),
    ])

    return transformTest

def ISIC_transform_discovery():

    transformDiscovery = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((512, 512))
        #transforms.Normalize((0,0,0),(1,1,1))
    ])

    return transformDiscovery