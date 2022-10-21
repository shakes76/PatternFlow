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

from torch.utils.data import Dataset
import os
from PIL import Image

# define dataset
class PixelCNNData(Dataset):
    def __init__(self, model, transforms):
        self.model = model
        self.images = os.listdir(TRAIN_PATH)
        self.tfs = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, x):
        img_path = TRAIN_PATH
        image = Image.open(img_path).convert('RGB')
        image = self.tfs(image)
        image = image.unsqueeze(dim=0)
        image = image.to(DEVICE)
        encoded_output = self.model.encoder(image)
        z = model.conv1(encoded_output)
        _,_,_,z = model.vq(z)
        z = z.float().to('cuda')
        z = z.view(64,64)
        z = torch.stack((z,z,z),0) # Pixel CNN uses 3 channel inputs
        return z,z

# create datasets and dataloaders
train_set = datasets.ImageFolder(root=root, transform=TRANSFORM)
test_set = datasets.ImageFolder(root=root, transform=TRANSFORM)
validate_set = datasets.ImageFolder(root=root, transform=TRANSFORM)

train_dl = DataLoader(train_set, batch_size=batch_size)
test_dl = DataLoader(test_set, batch_size=batch_size)
validate_dl = DataLoader(validate_set, batch_size=batch_size)



