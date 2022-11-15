import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)

class ISICDataset(Dataset):
    """
    PyTorch dataloader for the ISIC data set.
    
    Parameters:
        image_folder_path (str): path to a directory containing skin lesion images
        mask_folder_path (str): path to a directory containing mask images
    """
    def __init__(self, image_folder_path, mask_folder_path, diagnoses_path, device, transform=None):
        self.diagonoses_df = pd.read_csv(diagnoses_path)
        self.mask_folder_path = mask_folder_path
        self.transform = transform
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
            mask: UInt8Tensor[1, H, W]
            bounding box: Tensor[1, 4] bounding box generated from mask
        """
        image_id = self.diagonoses_df.iloc[idx]["image_id"]
        is_melanoma = self.diagonoses_df.iloc[idx]["melanoma"]
        if is_melanoma:
            label = 2
        else:
            label = 1
            
        # Load image
        # Network expects dim 2 in dim 0
        image_path = self.get_image_path(image_id)
        image = Image.open(image_path).convert("RGB")
        
        # Load mask image
        # Mask consists of two classes (including the background)
        mask_path = self.get_mask_path(image_id)
        mask = Image.open(mask_path)
        # if self.transform is not None:
        #    mask = self.transform(mask)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            assert xmin < xmax
            assert ymin < ymax
            assert xmin >= 0
            assert ymin >= 0
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([label], dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        
        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = labels
        targets["masks"] = masks
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, targets

if __name__ == "__main__":
    train_data = ISICDataset(
        image_folder_path="./data/ISIC-2017_Training_Data", 
        mask_folder_path="./data/ISIC-2017_Training_Part1_GroundTruth", 
        diagnoses_path="./data/ISIC-2017_Training_Part3_GroundTruth.csv",
        device=device,
        transform=get_transform(True),
        )
    
    image, target = train_data[0]
    fig, ax = plt.subplots()
    image = np.array(image.detach().cpu())
    ax.imshow(image.transpose((1,2,0)))
    bbox = target["boxes"][0]
    rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width()/8.0
    cy = ry - rect.get_height()/22.0
    label = "Melanoma" if target["labels"][0] == 2 else "Non-Melanoma"
    l = ax.annotate(
            label,
            (cx, cy),
            fontsize=7,
            # fontweight="bold",
            color="r",
            ha='center',
            va='center'
          )
    ax.add_patch(rect)
    fig, ax = plt.subplots()
    ax.imshow(target["masks"][0,...])
