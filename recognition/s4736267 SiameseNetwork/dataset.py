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
import PIL

#######################################################
#                  Crop Area ---- Not WORKING !!!!
#######################################################
#left, top, width, height   # Not WORKING !!!!

def crop_area_pos(loader):
    for i, data in enumerate(loader, 0):
        images= data[0]
        #print(images.shape)
        
        if i == 0:
            new = images[0] 
        else:
            new += images[0] 

    treshhold = 1
    print(new)
    test=new>treshhold
    torch.save(test, 'test.txt')
    print(test)

    print(torch.max(new))

    sum_total = torch.sum(new)
    sum_total = sum_total
    print(sum_total)

    sum_total = torch.sum(new*test)
    sum_total = sum_total
    print(sum_total)

    top=left=0
    height=240
    width=256
    
    for i in range(240):
        if torch.sum(transforms.functional.crop(new,i,0,240-i,256))<sum_total:
            print(transforms.functional.crop(new,i,0,240-i,256).shape)
            print("TOP", i)
            top=i
            break


    for i in range(256):
        if torch.sum(transforms.functional.crop(new,top,i,240-top,256-i))<sum_total:
            print(transforms.functional.crop(new,top,i,240-top,256-i).shape)
            print("LEFT", i)
            left=i
            break

       
    for i in range(240):
        if torch.sum(transforms.functional.crop(new,0,0,240-i,256))<sum_total:
            print(transforms.functional.crop(new,0,0,240-i,256).shape)
            print("Height", 240-i)
            height=240-i
            break
    

    for i in range(256):
        if torch.sum(transforms.functional.crop(new,0,0,240,256-i))<sum_total:
            print(transforms.functional.crop(new,0,0,240,256-i).shape)
            print("Width", 256-i)
            width=256-i

            break

    print(torch.sum(transforms.functional.crop(new,top,left,height,width))/sum_total)
    print(transforms.functional.crop(new,top,left,height,width).shape)        

    return 




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

    print("Mean of train_set",mean, flush=True)
    print("Std  of train_set",std, flush=True)


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


    image_paths_new=[None]*length
    image_paths_new2=[None]*length

###BACK Old versions, only shuffles sets
    #randompos = range(number_of_sets)
    #randompos = list(randompos)
    #random.shuffle(randompos)
    #print(len(image_paths_new))
#    i = 0 
#    while  i < number_of_sets:
#        placeholder=image_paths[0+i*20:20+i*20]
#        placeholder.sort(key=len, reverse=False)
#        j = int(randompos[i])
#        image_paths_new[0+j*20:20+j*20]=placeholder
#        i = i+1


    ###Sort Sets and Shuffle Set with other Sets

    randompos = range(number_of_sets)
    randompos = list(randompos)
    random.shuffle(randompos)
    
    i = 0 
    while  i < number_of_sets:
        placeholder=image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        j = int(randompos[i])
        image_paths_new[0+j*20:20+j*20]=placeholder
        i = i+1

    ###Shuffle Slice 1 to different set Slice 1
    #print("image_paths_new dataset.py",image_paths_new[:TRAIN_DATA_SIZE])

    for s in range(20):
        randompos2 = range(number_of_sets)
        randompos2 = list(randompos2)
        random.shuffle(randompos2)

    
        i=0
        while  i < number_of_sets:
            placeholder=image_paths_new[s+i*20]
            j = int(randompos2[i])
            #print("s:",s,"i:",i,"j:",j," - ", (s+j*20)%length)
            if image_paths_new2[(s+j*20)%length] != None:
                print("error: Place already taken train/valid_data() in dataset.py")
                
            else:
                image_paths_new2[(s+j*20)%length]=placeholder
            i = i+1


    #print("image_paths_new2 dataset.py",image_paths_new2[:TRAIN_DATA_SIZE])


    train_image_paths = image_paths_new2[:TRAIN_DATA_SIZE]
    valid_image_paths = image_paths_new2[length-VALID_DATA_SIZE:] 
    
    return train_image_paths, valid_image_paths

def test_data(test_data_path = 'ADNI_AD_NC_2D/AD_NC/test',DATA_SIZE = 20):
    
    image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        image_paths.append(glob.glob(data_path + '/*'))

    image_paths = list(flatten(image_paths))
    image_paths = sorted(image_paths)

    length = len(image_paths) 
    number_of_sets = int(length/20)

    image_paths_new=[None]*length
    image_paths_new2=[None]*length

    ###Sort Sets and Shuffle Set with other Sets
    randompos = range(number_of_sets)
    randompos = list(randompos)
    random.shuffle(randompos)
    
    i = 0 
    while  i < number_of_sets:
        placeholder=image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        j = int(randompos[i])
        image_paths_new[0+j*20:20+j*20]=placeholder
        i = i+1

    ###Shuffle Slice 1 to different set Slice 1

    for s in range(20):
        randompos2 = range(number_of_sets)
        randompos2 = list(randompos2)
        random.shuffle(randompos2)

    
        i=0
        while  i < number_of_sets:
            placeholder=image_paths_new[s+i*20]
            j = int(randompos2[i])
            #print("s:",s,"i:",i,"j:",j," - ", (s+j*20)%length)
            if image_paths_new2[(s+j*20)%length] != None:
                print("error: Place already taken test_data() in dataset.py")
                
            else:
                image_paths_new2[(s+j*20)%length]=placeholder
            i = i+1

    return image_paths_new[:DATA_SIZE]


#TODO give back images from filepaths / custom dataset or something  Maybe just use the first twenty entries of data loader ?
def clas_data(clas_data_path = 'ADNI_AD_NC_2D/AD_NC/test'):    
    
    image_paths = []

    for data_path in glob.glob(clas_data_path + '/*'):
        image_paths.append(glob.glob(data_path + '/*'))

    image_paths = list(flatten(image_paths))
    image_paths = sorted(image_paths)

    #image_paths.sort(key=lambda item: (-len(item), item))
    #image_paths = sorted(image_paths)
    
    length = len(image_paths) 
    number_of_sets = int(length/20)

    #image_paths.sort()
    image_paths_new=[None]*length

    i = 0 
    while  i < number_of_sets:
        placeholder=image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        image_paths_new[0+i*20:20+i*20]=placeholder
        i = i+1
    
    image_paths=image_paths_new[0:20] + image_paths_new[length-20:]
    
    #print("clas_data image_paths",image_paths)

    return image_paths


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

        slice_number = idx%20    
        
        #print(image_filepath)
        #print(label)
        #print(slice_number)

        return image, label, slice_number

class DatasetClas(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = len(image_paths)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        #print("DatasetClas image_filepath",image_filepath, " with index", idx)
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        
        label = image_filepath.split('/')[-2]
        label = 0 if label =='AD' else 1
        

        if self.transform:
            image = self.transform(image)
        
        slice_number = idx%20

        return image


class DatasetTrain(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = len(image_paths)         
        
    def __len__(self):
        return self.size

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
def clas_output(clas_dataset,slice_number,input_data):
    i=0
    clas_image_AD = torch.zeros_like(input_data)
    clas_image_NC = torch.zeros_like(input_data)

    for i in range(input_data.shape[0]):
        
        clas_image_AD[i] = clas_dataset[slice_number[i]]
        clas_image_NC[i] = clas_dataset[slice_number[i]+20] #Second Set represents NC


    return clas_image_AD, clas_image_NC

#######################################################
#                  Create Transformations
#######################################################    

# From https://medium.com/@sergei740/simple-guide-to-custom-pytorch-transformations-d6bdef5f8ba2
def crop(image: PIL.Image.Image) -> PIL.Image.Image:
    #left, top, width, height = 40-7+5-15-10, 10+5-15, (256-40-40+7+7-5-5+30), (240-10-40-5-5+30)
    left, top, width, height = 25, 5, 210, 210
    
    return transforms.functional.crop(image, top=top, left=left, height=height, width=width,)


def transformation_input():
    transform_input = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Lambda(crop),
        transforms.Resize(size=(105,105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1963, std=0.2540,inplace=True),
    ])

    return transform_input

def transformation_augmentation():
    transform_augmentation = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Lambda(crop),
        #transforms.RandomRotation((0,90)),
        #transforms.RandomResizedCrop(210, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.Resize(size=(105,105)),
        transforms.RandomCrop(105, padding=30,padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean=0.1963, std=0.2540,inplace=True),
    ])

    return transform_augmentation




def dataset(batch_size=64, TRAIN_SIZE = 200, VALID_SIZE= 20, TEST_SIZE=20):

    #Preparation
    
    train_image_paths, valid_image_paths = train_valid_data(TRAIN_DATA_SIZE = TRAIN_SIZE,VALID_DATA_SIZE = VALID_SIZE)
    test_image_paths = test_data(DATA_SIZE=TEST_SIZE)
    clas_image_paths = clas_data()

    #Dataset
    
    train_dataset = DatasetTrain(train_image_paths,transformation_augmentation())
    valid_dataset = DatasetTrain(valid_image_paths,transformation_input()) 
    test_dataset = Dataset(test_image_paths,transformation_input())
    clas_dataset = DatasetClas(clas_image_paths,transformation_input())


    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=1)

    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True,num_workers=1)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,num_workers=1)

    return train_loader, valid_loader, test_loader, clas_dataset

