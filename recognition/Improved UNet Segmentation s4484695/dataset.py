import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import math

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
from torchvision.io import read_image

# Hyper-Parameters can adjust these to affect loss and dice coefficients of training and test of model. 
# These parameters achieve the target goal of >0.8 dice coefficient average on test set.
imageReduction = 128
cropCoefficient = 0.9

# Ensure to specify transforms for DataSet or otherwise will default to no transforms applied.
class ISIC2017DataSet(Dataset):
    """
    Implements a class object representing ISIC2017DataSet inheriting from DataSet
    """
    def __init__(self, imgs_path, labels_path, transform=None, labelTransform=None):
        """
        Initialize class object with path and transforms. This is a mandatory function to implement a DataSet.

        imgs_path: path to folder containing images
        labels_path: path to folder containing label masks, labels and images must be in separate folders.
        transform: transforms to apply to images, of type torchvision.transforms
        labelTransform: transforms to apply to labels, of type torchvision.transforms
        """
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
        """
        This is a mandatory function to implement a DataSet.

        Calculates the length/size of the DataSet.
        """
        if self.ImagesSize != self.LabelsSize:
            print("Bad Data! Please Check Data, or unpredictable behaviour!")
            return -1
        else:
            return self.ImagesSize

    # Using os.path.join with self.imageNames + "_segmentation" ensures img_path and label_path both refer to same sample. 
    # This is required as os.listdir does not gaurantee order.
    def __getitem__(self, idx):
        """
        This is a mandatory function to implement a DataSet.

        Gets an image at index:idx and it's corresponding label. Label name must be in format of imageName_segmentation.png.

        idx: integer (index)
        return: tuple (image, label)
        """
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
    """
    Tranforms to images for training.

    return: transforms
    """
    transformTrain = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((imageReduction, imageReduction))
        #transforms.Normalize((0.0019, 0.0016, 0.0015), (0.0375, 0.0318, 0.0298))
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(math.sqrt(cropCoefficient*imageReduction*imageReduction))
    ])

    return transformTrain

def ISIC_transform_test():
    """
    Tranforms to images for test inference.

    return: transforms
    """
    transformTrain = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((imageReduction, imageReduction))
    ])

    return transformTrain

def ISIC_transform_label():
    """
    Tranforms to images for labels.

    return: transforms
    """
        
    transformTest = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((imageReduction, imageReduction))
    ])

    return transformTest

def ISIC_transform_discovery():
    """
    Tranforms to images for discovery (Calculating normalization values).

    return: transforms
    """

    transformDiscovery = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((imageReduction, imageReduction))
    ])

    return transformDiscovery