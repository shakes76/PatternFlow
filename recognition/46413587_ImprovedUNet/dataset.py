import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import argparse
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import os
import numpy as np
import matplotlib.pyplot as plt
import random

TrainImRoot="\DataSets\ISIC-2017_Training_Data"
TrainLbRoot="\DataSets\ISIC-2017_Training_Truth"
#TestImRoot="\DataSets\ISIC-2017_Test_Data"
TestLbRoot="\DataSets\ISIC-2017_Test_Truth"
ValImRoot="\DataSets\ISIC-2017_Validation_Data"
ValLbRoot="\DataSets\ISIC-2017_Validation_Truth"
workers = 2
batch_size=128
image_size=64
channels=3
num_epochs=20
learn_rate=0.0002
beta1=0.5

trainset=dset.ImageFolder(root = TrainImRoot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            ]))

dataloader=torch.utils.data.DataLoader(TrainImRoot, shuffle = True, batch_size=batch_size,
                                        numworkers = workers)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("no errors")