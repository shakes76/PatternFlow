import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
"""
All images returned are of size 3 * 240 * 240
"""

def transform():
    """
    Transform that crops some of than blank space out of the images. 
    """
    return transforms.Compose([transforms.Lambda(lambda x:  transforms.functional.crop(x, 0, 16, 240, 240)), 
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x: resize(x))])

def resize(x):
    """
    Transforms the images from 3x200x200 to 200x200x3
    """
    return torch.stack([x[i] for i in range(3)], dim=2)

def train_data_loader(dir, batch_size=32):
    train_dataset = datasets.ImageFolder(dir + '/train', transform=transform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def test_data_loader(dir, batch_size=32):
    test_dataset = datasets.ImageFolder(dir + '/test', transform=transform())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_dataloader