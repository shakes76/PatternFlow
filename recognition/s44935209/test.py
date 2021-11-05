from __future__ import print_function, division
import os
import numpy as np
from PIL import Image

import torch.nn.functional as F
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import torchsummary

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from Metrics import dice_coeff, accuracy_score
import time
#from ploting import VisdomLinePlotter
#from visdom import Visdom


#######################################################
#Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

batch_size = 4
print('batch_size = ' + str(batch_size))

valid_size = 0.15

epoch = 15
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
#Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary


model_test = model_unet(model_Inputs[0], 3, 1)

model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################

torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
#Passing the Dataset of Images and Labels
#######################################################


test_folderP = './ISIC-copy/ISIC2018_Task1-2_Training_Input_x2/'
test_folderL = './ISIC-copy/ISIC2018_Task1_Training_GroundTruth_x2/'

#######################################################
#Giving a transformation for input data
#######################################################

data_transform = torchvision.transforms.Compose([
           torchvision.transforms.Resize((128, 128)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

data_transform2 = torchvision.transforms.Compose([
           torchvision.transforms.Resize((128, 128)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])

#######################################################
#checking if cuda is available
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#######################################################
#Loading the model
#######################################################

model_test.load_state_dict(torch.load('./model/Unet_D_10_4/Unet_epoch_10_batchsize_4.pth'))

model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
#######################################################

# read_test_folder = glob.glob(test_folderP)
#
# x_sort_test = natsort.natsorted(read_test_folder)  # To sort
x_sort_test = [test_folderP + i for i in sorted(os.listdir(test_folderP))]

read_test_folder112 = './model/gen_images'


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

read_test_folder_P_Thres = './model/pred_threshold'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

read_test_folder_L_Thres = './model/label_threshold'


if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)
