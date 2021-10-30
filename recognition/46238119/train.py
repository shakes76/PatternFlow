"""
Author: Yaxian Shi
Student No.: 46238119
Training loop of StyleGAN2
"""

import random
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
import torchvision
from torchvision import datasets, transforms
import pickle as pkl
from tqdm import tqdm 
from torch.utils.data import Dataset
from model import *
import config

def MyDataloader():
  """
  Dataloader for dataset with DATA_ROOT.
  -- Resize train img into TRAIN_IMG_SIZE;
  -- Mormalize img;
  -- Apply data augmentation of RandomHorizontalFlip;
  -- Grayscale for output image with dimension 1, disable it if output image 
     with dimension greater than 1
  config need:
  -- DATA_ROOT, TRAIN_IMG_SIZE, IMG_DIMENSION, BATCH_SIZES, NUM_WORKERS, PIN_MEMORY
  """
  dataset = datasets.ImageFolder(
    root = config.DATA_ROOT,
    transform = transforms.Compose([
        transforms.Resize(config.TRAIN_IMG_SIZE),
        transforms.Grayscale(config.IMG_DIMENSION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5 for _ in range(config.IMG_DIMENSION)], 
                              std=[0.5 for _ in range(config.IMG_DIMENSION)]),
        transforms.RandomHorizontalFlip()
    ])
  )

  dataloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=config.BATCH_SIZES,
                                          num_workers=config.NUM_WORKERS,
                                          pin_memory=config.PIN_MEMORY,
                                          shuffle=True)
  
  return dataloader

def train():
  pass

def main():
  pass

if __name__ == "__main__":
  main()
  