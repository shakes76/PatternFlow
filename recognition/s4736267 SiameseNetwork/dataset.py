#dataset.py

#containing the data loader for loading and preprocessing your data.



import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import Dataset

def mean_std_calculation(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std
#otherway to load grayscale image
def image_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning 
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img

def train_set(batch_size=32):
	

	#With data augmentation
	transformation_train = transforms.Compose([		                                
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(105,105)),
		transforms.ToTensor(),
		transforms.Normalize(mean=0.1155, std=0.2224)
    ])

	#Define Training Set
	train_set = datasets.ImageFolder("~/COMP3710/Assignment3/ADNI_AD_NC_2D/AD_NC/train", transform = transformation_train)
	
	#Defining Dataloader
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

	return train_loader

def test_set(batch_size=32):
	
	#Testing without data augmentation
	transformation_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(105,105)),
		transforms.ToTensor(),
		transforms.Normalize(mean=0.1155, std=0.2224)
	])

	#Define Test Set
	test_set = datasets.ImageFolder("~/COMP3710/Assignment3/ADNI_AD_NC_2D/AD_NC/test", transform = transformation_test,loader=image_loader)
	
	#Defining Dataloader
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

	return test_loader


