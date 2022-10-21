import torch
from torchvision import datasets, transforms
"""
All images returned are of size 1 * 240 * 240
"""

def transform():
    """
    Transform that convert to grayscale, before applying a random crop and resizing to appropriate dimensions. 
    """
    return transforms.Compose([transforms.Lambda(lambda x:  transforms.functional.rgb_to_grayscale(x)),
                                transforms.RandomCrop(240),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x: resize(x))])

def resize(x):
    """
    Transforms the images from 1x240x240 to 240x240x1
    """
    return x[0][:, :, None]

def train_data_loader(dir, batch_size=32):
    train_dataset = datasets.ImageFolder(dir + '/train', transform=transform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def test_data_loader(dir, batch_size=32):
    test_dataset = datasets.ImageFolder(dir + '/test', transform=transform())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_dataloader