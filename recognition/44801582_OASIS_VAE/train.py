"""
Source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training.
"""
import os

import cv2
from torch.utils.data import Dataset
import torchvision


class OASISDataset(Dataset):
    def __init__(self, dir, dataset):
        # root = "keras_png_slices_data/keras_png_slices_"
        # if dataset == "train":
        #     self.data = os.listdir(root + "train")
        # elif dataset == "test":
        #     self.data = os.listdir(root + "test")
        # elif dataset == "validate":
        #     self.data = os.listdir(root + "validate")
        # else:
        #     print("Invalid type used, defaulting to train")
        #     self.data = os.listdir(root + "train")
        self.data = os.listdir(dir)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.data[index])
        image = self.transform(image)
        return image