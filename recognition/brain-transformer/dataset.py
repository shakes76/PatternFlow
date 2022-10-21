from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch

def torch_train(directory,batch_size=64,validation_split=0.2):
    '''Load training data into a Pytorch DataLoader that produce one hot encoded labels
    
    :param    directory : Directory to obtain training data
    :param    batch_size : Size of batches returned by DataLoader
    :param   validation_split : Proporation retained for validation

    Returns:
        DataLoader : Contains the training set
        DataLoader : Contains the validation set, if validation_split=0, Dataloader is empty
    '''
    transform = transforms.Compose([transforms.Grayscale(),transforms.Resize((256, 256)),transforms.ToTensor()])
    dataset = datasets.ImageFolder(directory,transform=transform, target_transform=lambda y : torch.zeros(2,dtype=torch.float).scatter_(0,torch.tensor(y), value=1))
    
    #Determine size of validation split 
    size = len(dataset)
    valid_size = int(np.floor(validation_split*size))

    #Seperate sample indices for validation and train split through sampler
    indicies = list(range(size))
    np.random.shuffle(indicies)
    train_indices, val_indices = indicies[valid_size:], indicies[:valid_size]
    tsampler = SubsetRandomSampler(train_indices)
    vsampler = SubsetRandomSampler(val_indices)

    #Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size,sampler=tsampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,sampler=vsampler)

    return train_loader, validation_loader

def torch_test(directory,batch_size=64):
    """ Load test data into a Pytorch DataLoader that outputs one hot encoded labels

    :param directory: directory of test set
    :param batch_size: batch size of dataloader to output

    Returns:
         DataLoader: dataloader containing test dataset
    
    """
    transform = transforms.Compose([transforms.Grayscale(),transforms.Resize((256, 256)),transforms.ToTensor()])
    dataset = datasets.ImageFolder(directory,transform=transform, target_transform=lambda y : torch.zeros(2,dtype=torch.float).scatter_(0,torch.tensor(y), value=1))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader