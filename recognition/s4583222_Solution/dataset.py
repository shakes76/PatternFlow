import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

training_images = '/home/Student/s4583222/COMP3710/Images/Train' #Directory of the training brain images
IMG_SIZE = 128 #Size to rescale the images to

class BrainDataset(Dataset):
    """
    Class to preprocess the Oasis Brain Dataset.
    We do not want to include any of the brain masks
    """
    def __init__(self, image_directory, transform = None):
        self.image_directory = image_directory
        self.transform = transform
        self.images = os.listdir(image_directory)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.images[index])
        image = Image.open(image_path).convert("RGB")#Open image as RGB

        if self.transform is not None:
            image = self.transform(image)#Perform transforms on the image

        return image

def get_data_loaders(training_images, batch, workers, pin_mem):
    """
    Given the training image path, batch size, number of workers, and pin memory status.
    This function creates the transforms for each image, transforms the image, and then passes it to DataLoader.
    The function then returns the data loader for the training images.
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),#Resize the images
        torchvision.transforms.RandomHorizontalFlip(),#Randomly flip image
        torchvision.transforms.ToTensor(),#Convert to a tensor
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1)#Normalise
    ])
    training_dataset = BrainDataset(image_directory=training_images, transform=transforms)#Transform image
    training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=workers, pin_memory=pin_mem, drop_last=True)#Pass transformed images to DataLoader
    return training_loader #Return the data loader for the training images.