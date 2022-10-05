#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import cv2

import glob
#from tqdm import tqdm

import random

#######################################################
#                  Mean and Standard Derivation Calculation
#######################################################


def mean_std_calculation(loader):
    mean = 0.
    std = 0.
    for i, data in enumerate(loader, 0):
        images= data[0]
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


#######################################################
#                  Data Preparation
####################################################### 

def train_valid_data(train_data_path = 'ADNI_AD_NC_2D/AD_NC/train',TRAIN_DATA_SIZE = 100,VALID_DATA_SIZE = 10):
    
    train_image_paths = [] #to store image paths in list
    for data_path in glob.glob(train_data_path + '/*'):
        #classes.append(data_path.split('/')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*'))
    
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    train_image_paths, valid_image_paths = train_image_paths[:TRAIN_DATA_SIZE],train_image_paths[TRAIN_DATA_SIZE:(TRAIN_DATA_SIZE+VALID_DATA_SIZE)] 
    #print(TRAIN_DATA_SIZE)
    return train_image_paths, valid_image_paths

def test_data(test_data_path = 'ADNI_AD_NC_2D/AD_NC/test',DATA_SIZE = 10):
    
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))
    random.shuffle(test_image_paths)

    return test_image_paths[:DATA_SIZE]


##########################################################



def classification_data(data_path = 'ADNI_AD_NC_2D/AD_NC/test'):
    
    image_path_AD = []
    image_path_NC = []

    image_path_AD.append(glob.glob(data_path + '/AD/*.jpeg')[0])    
    image_path_NC.append(glob.glob(data_path + '/NC/*.jpeg')[0])
    
    
    #image_path_AD = list(flatten(image_path_AD))
    #image_path_NC = list(flatten(image_path_NC))    

    image_NC = cv2.imread(image_path_NC[0], cv2.IMREAD_GRAYSCALE)
    image_AD = cv2.imread(image_path_AD[0], cv2.IMREAD_GRAYSCALE)

    transform_clas_data=trans_test()

    image_NC = transform_clas_data(image_NC)
    image_AD = transform_clas_data(image_AD)

    return image_NC, image_AD


#######################################################
#               Define Dataset Class
#######################################################

class Dataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        
        label = image_filepath.split('/')[-2]
        label = 0 if label =='AD' else 1
        

        if self.transform:
            image = self.transform(image)
        
        return image, label

class DatasetTrain(Dataset):
    def __init__(self, image_paths, transform=None, DATA_SIZE = 100):
        self.image_paths = image_paths
        self.transform = transform
        self.size = DATA_SIZE 
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx_1):
        image_filepath = self.image_paths[idx_1]
        image_1 = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        label_1 = image_filepath.split('/')[-2]
        label_1 = 0 if label_1=='AD' else 1
        
        idx_2 = random.randint(0,self.size-1)

        image_filepath = self.image_paths[idx_2]
        image_2 = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        label_2 = image_filepath.split('/')[-2]
        label_2 = 0 if label_2=='AD' else 1

        if label_1 == label_2:
            label=1
        else:
            label=0

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        return image_1, image_2, label

#######################################################
#                  Create Transformations
#######################################################    
def trans_train():
    transformation_train = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomCrop((105,105)),                                     
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Resize(size=(105,105)),
        transforms.crop(self,top=30,left=70,height=165,width=140)
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1143, std=0.2130)
    ])

    return transformation_train

def trans_valid():
    transformation_valid = transforms.Compose([
        transforms.ToPILImage(),                                     
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(105,105)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=0.000, std=1.000)
    ])

    return transformation_valid

def trans_test():
    transformation_test = transforms.Compose([
        transforms.ToPILImage(),                                     
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(105,105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1143, std=0.2130)
    ])

    return transformation_test


def dataset(batch_size=64, TRAIN_SIZE = 100, VALID_SIZE= 10, TEST_SIZE=10):

    #Preparation
    
    train_image_paths, valid_image_paths = train_valid_data(TRAIN_DATA_SIZE = TRAIN_SIZE,VALID_DATA_SIZE = VALID_SIZE)
    test_image_paths = test_data(DATA_SIZE=TEST_SIZE)

    transformation_train = trans_train()
    transformation_valid = trans_valid()
    transformation_test = trans_test()
    #Dataset
    
    train_dataset = DatasetTrain(train_image_paths,transformation_train,DATA_SIZE=TRAIN_SIZE)
    valid_dataset = DatasetTrain(valid_image_paths,transformation_valid,DATA_SIZE=VALID_SIZE) 
    test_dataset = Dataset(test_image_paths,transformation_test)

    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader