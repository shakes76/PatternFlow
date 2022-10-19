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

import cv2

import glob

import random
import PIL

from torchvision.utils import save_image

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

    for s in range(0):
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


    train_image_paths = image_paths_new[:TRAIN_DATA_SIZE]
    valid_image_paths = image_paths_new[length-VALID_DATA_SIZE:] 
    
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

    for s in range(0):
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


def clas_data(clas_data_path = 'ADNI_AD_NC_2D/AD_NC/test'):    
    
    image_paths = []

    for data_path in glob.glob(clas_data_path + '/*'):
        image_paths.append(glob.glob(data_path + '/*'))

    image_paths = list(flatten(image_paths))
    image_paths = sorted(image_paths)
    
    length = len(image_paths) 
    number_of_sets = int(length/20)

    image_paths_new=[None]*length

    i = 0 
    while  i < number_of_sets:
        placeholder=image_paths[0+i*20:20+i*20]
        placeholder.sort(key=len, reverse=False)
        image_paths_new[0+i*20:20+i*20]=placeholder
        i = i+1
    
    image_paths=image_paths_new[0:20*10] + image_paths_new[length-20*10:]
    return image_paths


#######################################################
#               Define Dataset Class
#######################################################

class DatasetTrain3D(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = int(len(image_paths)/20)    
    
    def transform_augmentation(self):

        transform_train=transformation_3D()

        #RandomCropResize
        rn = random.randint(0,1)
        if rn==1:
            left_off = random.randint(0,20)
            top_off  = random.randint(0,20)

            width_off = random.randint(0,20)
            height_off  = random.randint(0,20)

            transform_train.transforms.insert(3,Random_Crop_Resize(left_off=left_off,top_off=top_off,width_off=width_off,height_off=height_off))

        #RandomHorizontalFlip
        rn = random.randint(0,1)
        if rn==0:
            transform_train.transforms.insert(1,transforms.RandomHorizontalFlip(p=1))

        #RandomVerticalFlip
        rn = random.randint(0,1)
        if rn==0:
            transform_train.transforms.insert(1,transforms.RandomVerticalFlip(p=1))
        
        

        return transform_train


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        image3D_1=torch.zeros(20,210,210)
        image3D_2=torch.zeros(20,210,210)
        image3D_3=torch.zeros(20,210,210)

        idx_1=idx
        idx_2=(idx + random.randint(0,int(self.size)))%self.size
        idx_3=(idx + random.randint(0,int(self.size)))%self.size


        label_1 = self.image_paths[idx_1*20].split('/')[-2]
        label_1 = 0 if label_1=='AD' else 1
        
        label_2 = self.image_paths[idx_2*20].split('/')[-2]
        label_2 = 0 if label_2=='AD' else 1

        label_3 = self.image_paths[idx_3*20].split('/')[-2]
        label_3 = 0 if label_3=='AD' else 1


        while label_2 == label_3:
            idx_3=(idx + random.randint(0,int(self.size)))%self.size
            label_3 = self.image_paths[idx_3*20].split('/')[-2]
            label_3 = 0 if label_3=='AD' else 1    
            #print("new idx3",idx_3)

        #print(label_1,label_2,label_3)

        transform_idx_1=self.transform_augmentation()
        transform_idx_2=self.transform_augmentation()
        transform_idx_3=self.transform_augmentation()

        j=0
        for i in range(20):
            
            image_filepath_1 = self.image_paths[idx_1*20+i]
            image_1 = cv2.imread(image_filepath_1, cv2.IMREAD_GRAYSCALE)
            
            image_filepath_2 = self.image_paths[idx_2*20+i]
            image_2 = cv2.imread(image_filepath_2, cv2.IMREAD_GRAYSCALE)

            image_filepath_3 = self.image_paths[idx_3*20+i]
            image_3 = cv2.imread(image_filepath_3, cv2.IMREAD_GRAYSCALE)


            #toTen=transforms.ToTensor()
            #save_image(toTen(image_1), 'orignal.png')

            if self.transform:
                image_1 = transform_idx_1(image_1)
                image_2 = transform_idx_2(image_2)
                image_3 = transform_idx_3(image_3)

            #save_image(image_1, 'basic_augmentation.png')

            i  = 25
            for j in range(0):
                rn = random.randint(0,1)
                if rn==0:
                    
                    i1 = random.randint(0,210-i)
                    i2  = random.randint(0,210-i)
                
                    transform_blackout=Random_Blackout(i1=i1,i2=i2,i=i)
                    image_1=transform_blackout(image_1)
            
                rn = random.randint(0,1)
                if rn==0:
                    
                    i1 = random.randint(0,210-i)
                    i2 = random.randint(0,210-i)
                
                    transform_blackout=Random_Blackout(i1=i1,i2=i2,i=i)
                    image_2=transform_blackout(image_2)

                rn = random.randint(0,1)
                if rn==0:
                    
                    i1 = random.randint(0,210-i)
                    i2 = random.randint(0,210-i)
                
                    transform_blackout=Random_Blackout(i1=i1,i2=i2,i=i)
                    image_3=transform_blackout(image_3)
            
            #save_image(image_1, 'augmented.png')
        

            image3D_1[j]=image_1
            image3D_2[j]=image_2
            image3D_3[j]=image_3
            j=j+1

        if label_1==label_2:
            image3D_positive=image3D_2
            image3D_negative=image3D_3
            #print("two positive")
        else:
            image3D_positive=image3D_3
            image3D_negative=image3D_2
            #print("three positive")

        image3D_1 = torch.unsqueeze(image3D_1, dim=0)
        image3D_positive = torch.unsqueeze(image3D_positive, dim=0)
        image3D_negative = torch.unsqueeze(image3D_negative, dim=0)

        del transform_idx_1,transform_idx_2,transform_idx_3

        return image3D_1, image3D_positive, image3D_negative, label_1


class DatasetTrainClass(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = int(len(image_paths)/20)    
    
    def transform_augmentation(self):

        transform_train=transformation_3D()

        #RandomCropResize
        rn = random.randint(0,1)
        if rn==1:
            left_off = random.randint(0,20)
            top_off  = random.randint(0,20)

            width_off = random.randint(0,20)
            height_off  = random.randint(0,20)

            transform_train.transforms.insert(3,Random_Crop_Resize(left_off=left_off,top_off=top_off,width_off=width_off,height_off=height_off))

        #RandomHorizontalFlip
        rn = random.randint(0,1)
        if rn==0:
            transform_train.transforms.insert(1,transforms.RandomHorizontalFlip(p=1))

        #RandomVerticalFlip
        rn = random.randint(0,1)
        if rn==0:
            transform_train.transforms.insert(1,transforms.RandomVerticalFlip(p=1))
        
        

        return transform_train


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        image3D_1=torch.zeros(20,210,210)
        
        idx_1=idx
        
        label_1 = self.image_paths[idx_1*20].split('/')[-2]
        label_1 = 0 if label_1=='AD' else 1
        

        transform_idx_1=self.transform_augmentation()

        j=0
        for i in range(20):
            
            image_filepath_1 = self.image_paths[idx_1*20+i]
            image_1 = cv2.imread(image_filepath_1, cv2.IMREAD_GRAYSCALE)
            

            #toTen=transforms.ToTensor()
            #save_image(toTen(image_1), 'orignal.png')

            if self.transform:
                image_1 = transform_idx_1(image_1)

            #save_image(image_1, 'basic_augmentation.png')

            i  = 25
            for j in range(0):
                rn = random.randint(0,1)
                if rn==0:
                    
                    i1 = random.randint(0,210-i)
                    i2  = random.randint(0,210-i)
                
                    transform_blackout=Random_Blackout(i1=i1,i2=i2,i=i)
                    image_1=transform_blackout(image_1)
            
                rn = random.randint(0,1)
            
            #save_image(image_1, 'augmented.png')
        

            image3D_1[j]=image_1
            j=j+1

        image3D_1 = torch.unsqueeze(image3D_1, dim=0)

        del transform_idx_1

        return image3D_1, label_1






class Dataset3D(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = int(len(image_paths)/20)    
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        image3D=torch.zeros(20,210,210)


        j=0
        for i in range(20):
    
            image_filepath = self.image_paths[idx*20+i]
            image_1 = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
            
            if self.transform:
                image_1 = self.transform(image_1)

            image3D[j]=image_1

            j=j+1

        label = image_filepath.split('/')[-2]
        label = 0 if label=='AD' else 1
        

        image3D = torch.unsqueeze(image3D, dim=0)

        #print("idx:",idx,"  --  ",image_filepath,"  --  label:",label)

        return image3D, label

class DatasetClas3D(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.size = int(len(image_paths)/20)    
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        image3D=torch.zeros(20,210,210)

        j=0
        for i in range(20):
    
            image_filepath = self.image_paths[idx*20+i]
            image_1 = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
            
            if self.transform:
                image_1 = self.transform(image_1)

            image3D[j]=image_1

            j=j+1

        image3D = torch.unsqueeze(image3D, dim=0)    

        return image3D


#######################################################
#                  Return Classification Source
#######################################################  

def clas_output3D(clas_dataset,input_data):

    clas_Person_AD = torch.zeros_like(input_data)
    clas_Person_NC = torch.zeros_like(input_data)

    

    for i in range(input_data.shape[0]):
        clas_Person_AD[i] = clas_dataset[i]
        clas_Person_NC[i] = clas_dataset[i+10] #Second Set represents NC

    return clas_Person_AD, clas_Person_NC


#######################################################
#                  Create Transformations
#######################################################    

# From https://medium.com/@sergei740/simple-guide-to-custom-pytorch-transformations-d6bdef5f8ba2
def crop(image: PIL.Image.Image) -> PIL.Image.Image:
    #left, top, width, height = 40-7+5-15-10, 10+5-15, (256-40-40+7+7-5-5+30), (240-10-40-5-5+30)
    left, top, width, height = 25, 5, 210, 210
    
    return transforms.functional.crop(image, top=top, left=left, height=height, width=width,)

class Random_Crop_Resize:
    def __init__(self,left_off=0,top_off=0,width_off=0,height_off=0):
        self.left_off = left_off
        self.top_off = top_off
        self.width_off = width_off
        self.height_off = height_off
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        left, top, width, height = 25, 5, 210, 210
        return transforms.Resize((210,210))(transforms.functional.crop(img, top=top+self.top_off, left=left+self.left_off, height=height-self.height_off-self.top_off, width=width-self.width_off-self.left_off))

class Random_Blackout:
    def __init__(self,i1,i2,i):
        self.i1 = i1  #Pixel value
        self.i2 = i2  #Pixel value
        self.i = i      #Size of BlackoutArea
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return transforms.functional.erase(img, self.i1, self.i2, self.i, self.i, 0)


def transformation_3D():
    transform_input = transforms.Compose([
        
        transforms.ToPILImage(),
        transforms.Lambda(crop),
        transforms.ToTensor(),
        #transforms.Resize(size=(105,105)),
        #transforms.Normalize(mean=0.1963, std=0.2540,inplace=True),
    ])

    return transform_input

#######################################################
#                  Create Dataloader
####################################################### 


def dataset3D(batch_size=64, TRAIN_SIZE = 200, VALID_SIZE= 20, TEST_SIZE=20):

    #Preparation
    
    train_image_paths, valid_image_paths = train_valid_data(TRAIN_DATA_SIZE = TRAIN_SIZE,VALID_DATA_SIZE = VALID_SIZE)
    test_image_paths = test_data(DATA_SIZE=TEST_SIZE)
    clas_image_paths = clas_data()

    #Dataset
    
    train_dataset = DatasetTrain3D(train_image_paths,transformation_3D())
    valid_dataset = DatasetTrain3D(valid_image_paths,transformation_3D()) 
    test_dataset = Dataset3D(test_image_paths,transformation_3D())
    clas_dataset = DatasetClas3D(clas_image_paths,transformation_3D())

    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=1)

    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True,num_workers=1)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,num_workers=1)

    return train_loader, valid_loader, test_loader, clas_dataset


def datasetClass(batch_size=64, TRAIN_SIZE = 200, VALID_SIZE= 20, TEST_SIZE=20):

    #Preparation
    
    train_image_paths, valid_image_paths = train_valid_data(TRAIN_DATA_SIZE = TRAIN_SIZE,VALID_DATA_SIZE = VALID_SIZE)
    test_image_paths = test_data(DATA_SIZE=TEST_SIZE)

    #Dataset
    
    test_dataset = Dataset3D(test_image_paths,transformation_3D())

    train_dataset = DatasetTrainClass(train_image_paths,transformation_3D())
    valid_dataset = DatasetTrainClass(valid_image_paths,transformation_3D()) 

    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=1)

    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True,num_workers=1)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,num_workers=1)

    return train_loader, valid_loader, test_loader
