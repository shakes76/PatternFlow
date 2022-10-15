import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, img_folder_path, mask_folder_path):
        self.img_paths = glob.glob(os.path.join(img_folder_path, "*.jpg"))
        self.img_paths = [img_path for img_path in self.img_paths if "superpixels" not in img_path]
        self.mask_paths = glob.glob(os.path.join(mask_folder_path, "*.png"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        return torch.from_numpy(img), torch.from_numpy(mask)
        
    