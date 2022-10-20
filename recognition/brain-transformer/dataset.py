from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch

def torch_train(directory,batch_size=64,validation_split=0.2):
    transform = transforms.Compose([transforms.Grayscale(),transforms.Resize((256, 256)),transforms.ToTensor()])
    # targetTransform = transforms.Compose([one_hot])
    dataset = datasets.ImageFolder(directory,transform=transform, target_transform=lambda y : torch.zeros(2,dtype=torch.float).scatter_(0,torch.tensor(y), value=1))
    size = len(dataset)
    valid_size = int(np.floor(validation_split*size))
    indicies = list(range(size))
    np.random.shuffle(indicies)
    train_indices, val_indices = indicies[valid_size:], indicies[:valid_size]
    tsampler = SubsetRandomSampler(train_indices)
    vsampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size,sampler=tsampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,sampler=vsampler)
    return train_loader, validation_loader

def torch_test(directory,batch_size=64):
    transform = transforms.Compose([transforms.Grayscale(),transforms.Resize((256, 256)),transforms.ToTensor()])
    # targetTransform = transforms.Compose([one_hot])
    dataset = datasets.ImageFolder(directory,transform=transform, target_transform=lambda y : torch.zeros(2,dtype=torch.float).scatter_(0,torch.tensor(y), value=1))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader