# COMP3710 Pattern Recognition Lab Assignment
# By Thomas Jellett (s4583222)
# HARD DIFFICULTY
# Create a generative model of the OASIS brain using stable diffusion that
# has a â€œreasonably clear image.â€

# File: dataset.py
# Description: Used to get the OASIS brain dataset

import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader

training_images = '/home/Student/s4583222/COMP3710/Images/Train'
# Train_image_path = 's4583222_Solution/keras_png_slices_train/'
# Validate_image_path = 's4583222_Solution/keras_png_slices_validate/'
# Test_image_path = 's4583222_Solution/keras_png_slices_test/'

#Class for Brain Dataset. We do not care about the masked images
class BrainDataset(Dataset):
    def __init__(self, image_directory, transform = None):
        self.image_directory = image_directory
        self.transform = transform
        self.images = os.listdir(image_directory)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.images[index]) #Masks have same name as images
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image)
            # augmentations = self.transform(image=image)
            # image = augmentations["image"]

        return image

def get_data_loaders(training_images, batch, workers, pin_mem):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    training_dataset = BrainDataset(image_directory=training_images, transform=transforms)
    training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=workers, pin_memory=pin_mem)
    return training_loader

# training_loader = get_data_loaders(training_images, 1, 2, True)
# print(len(training_loader))
