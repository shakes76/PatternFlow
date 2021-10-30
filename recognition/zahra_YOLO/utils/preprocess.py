#This script reads all the images and resizes the image without changing the aspect ratio.
#in this script we adjust the size of the image by adding padding to the top, bottom, left, and right of the image
import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.read().splitlines() #file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 1

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)
        #img.size[0] = width
        width, height = img.size
        img_size=416  
        max_v = max(height, width)
        ratio = img_size / max_v
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        
        if imh>imw: 
            dim_diff =imh - imw
            pad_left, pad_right = dim_diff // 2, dim_diff - dim_diff // 2
            pad=(pad_left,0,pad_right,0)
        elif imw>imh:
            dim_diff =imw - imh
            pad_top, pad_bottom = dim_diff // 2, dim_diff - dim_diff // 2
            pad=(0,pad_top,0,pad_bottom)
    
 #this is the padding for the left, top, right and bottom borders respectively.
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad(pad,0),
         transforms.ToTensor(),
         ])
        

    # convert image to Tensor
        image_tensor = img_transforms(img).float()
    # convert image to Tensor
        input_img = img_transforms(img).float()

        #---------
        #  Label
        #---------
          
    
# this script extract the coordinates of the bounding box we created in the annotation file for the original image which is unpadded
#and unscaled, then updates the left-bottom and top-right coordinate of the bounding box according to the scaled and padded image

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        # _,padded_height, padded_width = input_img.shape
        
        annotation = None
        if os.path.exists(label_path):
            annotation = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = width * (annotation[:, 1] - annotation[:, 3]/2)
            y1 = height * (annotation[:, 2] - annotation[:, 4]/2)
            x2 = width * (annotation[:, 1] + annotation[:, 3]/2)
            y2 = height * (annotation[:, 2] + annotation[:, 4]/2)
            
            x1 += pad[0]
            y1 += pad[3]
            x2 += pad[2]
            y2 += pad[1]
            
            # Calculate ratios from coordinates
            annotation[:, 1] = ((x1 + x2) / 2) / max_v
            annotation[:, 2] = ((y1 + y2) / 2) / max_v
            annotation[:, 3] *= width / max_v
            annotation[:, 4] *= height / max_v
        
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if annotation is not None:
            filled_labels[range(len(annotation))[:self.max_objects]] = annotation[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)