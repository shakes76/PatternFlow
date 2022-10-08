
from vit_pytorch import ViT
import argparse
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import  DataLoader

from dataset import BrainDataset



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.manual_seed(42)
train_dir = './datasets/AD_NC/train/AD'


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=12)
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--heads', type=int, default=12)
parser.add_argument('--mlp_dim', type=int, default=512)
opt = parser.parse_args()


image_size = 224

train_tfm = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    # transforms.RandomGrayscale(p=0.2),
    #   transforms.RandomInvert(p=0.2),
    # # transforms.RandomPosterize(bits=2),
    # transforms.RandomSolarize(threshold=192.0, p=0.2),
    # transforms.RandomEqualize(p=0.2),
    transforms.ToTensor(),
    # transforms.Normalize(mean=0.00034013,std=(0.00064943))
    transforms.Normalize(mean=(0.2641372), std=(0.5060895))
])
test_tfm = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5), std=(0.5))
    # transforms.Normalize(mean=0.522844,std=(0.98886025))

    transforms.Normalize(mean=(0.2641372), std=(0.5060895))
])
batch_size = 64
_dataset_dir = "./datasets/AD_NC/"


train_set = BrainDataset(os.path.join(
    _dataset_dir, "train"), tfm=train_tfm, split='train')
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

val_set = BrainDataset(os.path.join(
    _dataset_dir, "train"), tfm=test_tfm, split='val')
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True)

test_set=BrainDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
test_loader=DataLoader(test_set, batch_size=batch_size,
                         shuffle=True, pin_memory=True)

print(len(train_set))