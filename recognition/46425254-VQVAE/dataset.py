import pandas as pd
import torch
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms




"""
DataLoader class that converts OASIS brain dataset into a meaningful format for 
training using VQVAE.

Paramaters:
    data_path -> a path to the training, testing
    or validation folder of OASIS brain images

"""
class DataLoader():
    """
    In order to make the dataset useful to pytorch functions, it is needed to
    process the dataset (images) into a readable format
    
    This class paramaterizes the training, testing and validation datasets and
    turns them into tensors
    
    """
    def __init__(self, data_path):
        
        data = torchvision.ImageFolder(root = data_path, transform 
                                        = transforms.ToTensor())
                

    
        self.data_loader = utils.data.DataLoader(data, batch_size = 256, 
                                                  num_workers = 1)
                

    #returns the data_loader object
    def __getitem__(self):
        return self.data_loader
    
    
    
    