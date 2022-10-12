# COMP3710 Pattern Recognition Lab Assignment
# By Thomas Jellett (s4583222)
# HARD DIFFICULTY
# Create a generative model of the OASIS brain using stable diffusion that
# has a “reasonably clear image.”

# File: predict.py
# Description: Loads in a trained model and then generates images from it

import os
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torchvision import transforms 
from dataset import get_data_loaders
from modules import Diffusion
from modules import UNETModel
import numpy as np
from PIL import Image
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10 #If we drop this to 1, then len(training_loader) = 11,328
IMG_HEIGHT = 256 #Actual is 256
IMG_WIDTH = 256 #Actual is 256
TRAINING_DIR = '/home/Student/s4583222/COMP3710/Images/Train'

def predict():
    model = UNETModel().to(DEVICE)
    model.load_state_dict(torch.load("Model_2.pt"))

if __name__ == '__main__':
    predict()
