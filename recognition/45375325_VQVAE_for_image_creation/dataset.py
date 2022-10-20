import torch
import torchvision.transforms as tf
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image

# defining constants
root = os.getcwd() + "/data"
TRAIN_PATH = root + "/train"
TEST_PATH = root + "/test"
VALIDATE_PATH = root + "/validate"
batch_size = 32
DEVICE = "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM = tf.Compose([
    tf.ToTensor()
])
codebook_transform = tf.Compose([
    tf.Resize(128),
    tf.ToTensor()
])


class NumpyDataset(Dataset):
    """
    Creates a dataset using numpy arrays for making use of the codebooks produced by the VQVAE
    """
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# create datasets and dataloaders
train_set = datasets.ImageFolder(root=root, transform=TRANSFORM)
test_set = datasets.ImageFolder(root=root, transform=TRANSFORM)
validate_set = datasets.ImageFolder(root=root, transform=TRANSFORM)

train_dl = DataLoader(train_set, batch_size=batch_size)
test_dl = DataLoader(test_set, batch_size=batch_size)
validate_dl = DataLoader(validate_set, batch_size=batch_size)
