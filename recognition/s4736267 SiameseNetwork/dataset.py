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

def train_valid_data(train_data_path = 'ADNI_AD_NC_2D/AD_NC/train',TRAIN_DATA_SIZE = 20,VALID_DATA_SIZE = 20):
    
    image_paths = [] #to store image paths in list

    for data_path in glob.glob(train_data_path + '/*'):
        #classes.append(data_path.split('/')[-1]) 
        image_paths.append(glob.glob(data_path + '/*'))
    
    image_paths = list(flatten(image_paths))
    image_paths = sorted(image_paths)

    length = len(image_paths)
    number_of_sets = int(length/20)

    randompos = range(number_of_sets)
    randompos = list(randompos)
    random.shuffle(randompos)

    image_paths_new=[None]*length
    #print(len(image_paths_new))
    i = 0 
    while  i < number_of_sets:
        placeholder=image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        j = int(randompos[i])
        image_paths_new[0+j*20:20+j*20]=placeholder
        i = i+1
        #print("i",i," ",j)
        
    #print(len(image_paths_new))
    #print("")
    #print(len(image_paths))
    #print("")
    image_paths = image_paths_new
    #print(all(v for v in image_paths_new))
    #print("len image_path", len(image_paths))

    train_image_paths, valid_image_paths = image_paths[:TRAIN_DATA_SIZE],image_paths[TRAIN_DATA_SIZE:(TRAIN_DATA_SIZE+VALID_DATA_SIZE)] 

    #print("len train_image_path", len(train_image_paths))
    return train_image_paths, valid_image_paths

def test_data(test_data_path = 'ADNI_AD_NC_2D/AD_NC/test',DATA_SIZE = 20):
    
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))
    test_image_paths = sorted(test_image_paths)

    length = len(test_image_paths) 
    number_of_sets = int(length/20)

    test_image_paths.sort()

    i = 0 
    while  i < number_of_sets:
        placeholder=test_image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        test_image_paths[0+i*20:20+i*20]=placeholder
        i = i+1
    
    return test_image_paths[:DATA_SIZE]


#TODO give back images from filepaths / custom dataset or something  Maybe just use the first twenty entries of data loader ?
def classification_data(class_data_path = 'ADNI_AD_NC_2D/AD_NC/test'):    
    
    image_paths = []

    for data_path in glob.glob(class_data_path + '/*'):
        image_paths.append(glob.glob(data_path + '/*'))

    image_paths = list(flatten(image_paths))
    
    #image_paths.sort(key=lambda item: (-len(item), item))
    #image_paths = sorted(image_paths)
    
    length = len(image_paths) 
    number_of_sets = int(length/20)

    image_paths.sort()
    
    i = 0 
    while  i < number_of_sets:
        placeholder=image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        image_paths[0+i*20:20+i*20]=placeholder
        i = i+1
       
    image_paths_AD = image_paths[0:20]        #First Set should be AD
    #print(len(image_paths_AD))
    image_paths_NC = image_paths[length-20:]  #Last Set should be NC
        
    print("")
    print(image_paths_AD)
    print("")
    print(image_paths_NC)

    return image_paths_AD, image_paths_NC


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
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = len(image_paths)         
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx_1):
        #print("path_size",self.size)
        image_filepath = self.image_paths[idx_1]
        #print(image_filepath, "-", idx_1)
        image_1 = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        label_1 = image_filepath.split('/')[-2]
        label_1 = 0 if label_1=='AD' else 1
        
        idx_help = idx_1 + random.randint(0,int(self.size/20))*20

        idx_2 = (idx_help) % self.size   #Assumption Slice index _ ist only important on number

        #print("idx_1",idx_1,"  idx_2",idx_2,"  idx_help",idx_help)

        image_filepath = self.image_paths[idx_2]
        #print(image_filepath, "-", idx_2)
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
        
        #print("Label:", label)

        #print("")
        return image_1, image_2, label

#######################################################
#                  Create Transformations
#######################################################    
def trans_train():
    transformation_train = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomCrop((105,105)),                                     
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(105,105)),
        #transforms.crop(self,top=30,left=70,height=165,width=140)
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


def dataset(batch_size=64, TRAIN_SIZE = 200, VALID_SIZE= 20, TEST_SIZE=20):

    #Preparation
    
    train_image_paths, valid_image_paths = train_valid_data(TRAIN_DATA_SIZE = TRAIN_SIZE,VALID_DATA_SIZE = VALID_SIZE)
    test_image_paths = test_data(DATA_SIZE=TEST_SIZE)

    transformation_train = trans_train()
    transformation_valid = trans_valid()
    transformation_test = trans_test()
    #Dataset
    
    train_dataset = DatasetTrain(train_image_paths,transformation_train)
    valid_dataset = DatasetTrain(valid_image_paths,transformation_valid) 
    test_dataset = Dataset(test_image_paths,transformation_test)

    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader