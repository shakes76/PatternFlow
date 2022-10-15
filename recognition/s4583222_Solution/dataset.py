import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader

training_images = '/home/Student/s4583222/COMP3710/Images/Train'
IMG_SIZE = 128

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
        # image = np.array(Image.open(image_path).convert("RGB"))
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

def get_data_loaders(training_images, batch, workers, pin_mem):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE,IMG_SIZE)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    training_dataset = BrainDataset(image_directory=training_images, transform=transforms)
    training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=workers, pin_memory=pin_mem, drop_last=True)
    return training_loader