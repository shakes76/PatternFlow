# COMP3710 Pattern Recognition Lab Assignment
# By Thomas Jellett (s4583222)
# HARD DIFFICULTY
# Create a generative model of the OASIS brain using stable diffusion that
# has a Ã¢â‚¬Å“reasonably clear image.Ã¢â‚¬Â

# File: train.py
# Description: Used to train the model

import os
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torchvision import transforms 
from dataset import get_data_loaders
from modules import UNETModel
import numpy as np
from PIL import Image
import torch.nn.functional as F

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 #If we drop this to 1, then len(training_loader) = 11,328
NUM_EPOCHS = 1000
NUM_WORKERS = 0
IMG_HEIGHT = 128 #Actual is 256
IMG_WIDTH = 128 #Actual is 256
PIN_MEMORY = False
LOAD_MODEL = False
TRAINING_DIR = '/home/Student/s4583222/COMP3710/Images/Train'

NOISE_STEP = 600

BETA = torch.linspace(0.0001, 0.02, NOISE_STEP)
ALPHA = 1. - BETA
ALPHA_HAT = torch.cumprod(ALPHA, axis=0) #alpha hat is the cumulative product of alpha
ALPHA_HAT_PREV = F.pad(ALPHA_HAT[:-1], (1,0), value=1.0)
SQRT_ALPHA_REC = torch.sqrt(1.0/ALPHA)
PROS_VAR = BETA * (1. - ALPHA_HAT_PREV)/(1. - ALPHA_HAT)

def index_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1,t.cpu())
    out = out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)
    return out

def forward_diffusion_to_image(image, t, device =DEVICE):
    # print(image.shape)
    sqrt_alpha_hat = index_list(torch.sqrt(ALPHA_HAT), t, image.shape)
    # sqrt_alpha_hat = sqrt_alpha_hat[:, :, :, None]
    # print(sqrt_alpha_hat.shape)
    sqrt_one_minus_alpha_hat = index_list(torch.sqrt(1. - ALPHA_HAT), t, image.shape)
    # print(sqrt_one_minus_alpha_hat.shape)
    epsilon = torch.randn_like(image)
    # print(epsilon.shape)

    # print(f"Epoch: {(sqrt_alpha_hat * image).shape[0]} | iteration: {(sqrt_one_minus_alpha_hat * epsilon).shape[0]}")       
    x_1 = sqrt_alpha_hat.to(device) * image.to(device)
    x_2 = sqrt_one_minus_alpha_hat.to(device) * epsilon.to(device)
    return sqrt_alpha_hat.to(device) * image.to(device) + sqrt_one_minus_alpha_hat.to(device) * epsilon.to(device), epsilon.to(device)

def model_loss(model, image, t):
    noisy_image, actual_noise = forward_diffusion_to_image(image, t)
    #show_image(noisy_image.detach().cpu(), 0, 0)
    predicted_noise = model(noisy_image, t)
    loss = F.l1_loss(actual_noise, predicted_noise)
    return loss
   
def training(): 
    training_loader = get_data_loaders(TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    model = UNETModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    if LOAD_MODEL:
        model.load_state_dict(torch.load("Model_2.pt"))

    vloss = []
    
    #Training Loop
    for epoch in range(NUM_EPOCHS):
        for i, images in enumerate(tqdm(training_loader)):          
            optimizer.zero_grad()
        
            time = torch.randint(0, NOISE_STEP, (BATCH_SIZE,), device=DEVICE).long()
            loss = model_loss(model, images, time)
            vloss.append(loss.item())            
        
            loss.backward()
            optimizer.step()

            #Save the model every 10 iterations in each epoch
            if epoch % 1 == 0 and i % 10 == 0:
                torch.save(model.state_dict(), "Model_2.pt")
                plot_v = torch.tensor(vloss, device='cpu')
                plt.title('Stable Diffusion Loss')
                plt.plot(plot_v, color = 'blue')
                plt.savefig('loss_plot')

            if epoch % 5 == 0 and i == 1: #Creates back up model in case other model saves incorrectly
                #sample_image(epoch, i, model)
                print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
                torch.save(model.state_dict(), "Model_backup.pt")
                
if __name__ == '__main__':
    training()