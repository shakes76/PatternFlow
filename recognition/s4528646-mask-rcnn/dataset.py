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
import cv2

class ISICDataset(Dataset):
    """
    PyTorch dataloader for the ISIC data set.
    
    Parameters:
        image_folder_path (str): path to a directory containing skin lesion images
        mask_folder_path (str): path to a directory containing mask images
    """
    def __init__(self, image_folder_path, mask_folder_path, diagnoses_path, device):
        self.diagonoses_df = pd.read_csv(diagnoses_path)
        self.mask_folder_path = mask_folder_path
        self.image_folder_path = image_folder_path
        self.device = device
        
    def get_image_path(self, image_id):
        return os.path.join(self.image_folder_path, image_id + ".jpg")
    
    def get_mask_path(self, image_id):
        return os.path.join(self.mask_folder_path, image_id + "_segmentation.png")
    
    def __len__(self):
        return self.diagonoses_df.shape[0]
    
    def __getitem__(self, idx):
        """
        Load a single sample of the ISIC dataset.
        Returns a tuple containing:
            image: Tensor[C, H, W] normalised to 0-1
            label: diagnoses label, 0 if no diagnosis, 1 if melanoma, 2 if seborrheic keratosis
            mask: UInt8Tensor[N, H, W]
            bounding box: Tensor[N, 4] bounding box generated from mask
        """
        image_id = self.diagonoses_df.iloc[idx]["image_id"]
        melanoma = self.diagonoses_df.iloc[idx]["melanoma"]
        seborrheic_keratosis = self.diagonoses_df.iloc[idx]["seborrheic_keratosis"]
        if melanoma:
            label = 1
        elif seborrheic_keratosis:
            label = 2
        else:
            label = 0
            
        """Load image"""
        image_path = self.get_image_path(image_id)
        image = cv2.imread(image_path) / 255.0
        image = np.swapaxes(image, 0, 2)
        
        """Load ground truth segmentation"""
        mask_path = self.get_mask_path(image_id)
        mask = cv2.imread(mask_path)
        mask = (mask > 0).astype(np.uint8)[..., 0]
        mask = np.expand_dims(mask, axis=0)
        
        """Compute bounding box"""
        x, y, w, h = cv2.boundingRect(mask[0, ...])
        
        targets = {
            "labels": torch.tensor([label], dtype=torch.int64), 
            "masks": torch.from_numpy(mask), 
            "boxes": torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
            }
        return torch.from_numpy(image).double(), targets
        