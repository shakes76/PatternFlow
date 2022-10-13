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
BATCH_SIZE = 128 #If we drop this to 1, then len(training_loader) = 11,328
NUM_EPOCHS = 1000
NUM_WORKERS = 0
IMG_HEIGHT = 64 #Actual is 256
IMG_WIDTH = 64 #Actual is 256
PIN_MEMORY = False
LOAD_MODEL = False
TRAINING_DIR = '/home/Student/s4583222/COMP3710/Images/Train'

NOISE_STEP = 300

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
    print(image.shape)
    sqrt_alpha_hat = index_list(torch.sqrt(ALPHA_HAT), t, image.shape)
    print(sqrt_alpha_hat.shape)
    sqrt_one_minus_alpha_hat = index_list(torch.sqrt(1. - ALPHA_HAT), t, image.shape)
    print(sqrt_one_minus_alpha_hat.shape)
    epsilon = torch.randn_like(image)
    print(epsilon.shape)

    # print(f"Epoch: {(sqrt_alpha_hat * image).shape[0]} | iteration: {(sqrt_one_minus_alpha_hat * epsilon).shape[0]}")       
    x_1 = sqrt_alpha_hat.to(device) * image.to(device)
    x_2 = sqrt_one_minus_alpha_hat.to(device) * epsilon.to(device)
    return sqrt_alpha_hat.to(device) * image.to(device) + sqrt_one_minus_alpha_hat.to(device) * epsilon.to(device), epsilon.to(device)

def show_image(image, epoch, i):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    plt.savefig(f"{epoch}_{i}")

def model_loss(model, image, t):
    noisy_image, actual_noise = forward_diffusion_to_image(image, t)
    show_image(noisy_image, 0, 0)
    predicted_noise = model(noisy_image, t)
    loss = F.l1_loss(actual_noise, predicted_noise)
    return loss

#--------------Predict stuff-------------------------#

    

@torch.no_grad()
def sample_image_at_timestep(rand_noise, t, model):

    beta_time = index_list(BETA, t, rand_noise.shape)
    sqrt_one_minus_alpha_hat_time = index_list(torch.sqrt(1. - ALPHA_HAT), t, rand_noise.shape)
    sqrt_alpha_reciprical_time = index_list(SQRT_ALPHA_REC, t, rand_noise.shape)
    prosterior_var_time = index_list(PROS_VAR, t, rand_noise.shape)

    predicted_noise = model(rand_noise, t)
    mean = sqrt_alpha_reciprical_time * (rand_noise - beta_time * predicted_noise / sqrt_one_minus_alpha_hat_time)
    if t == 0:
        return mean
    else:
        noise = torch.randn_like(rand_noise)
        return mean + torch.sqrt(prosterior_var_time) * noise  

@torch.no_grad()
def sample_image(epoch, i, model):
    # for num in range(1,num_images_to_gen):
    noise = torch.randn((1,3, IMG_HEIGHT, IMG_WIDTH), device=DEVICE)   
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 1
    stepsize = int(NOISE_STEP/num_images)
    for ind in range(0, NOISE_STEP)[::-1]:
        t = torch.full((1,), ind, device=DEVICE, dtype=torch.long)
        image = sample_image_at_timestep(noise, t, model)
        # if ind % stepsize == 0:
        #     plt.subplot(1, num_images, int(ind/stepsize)+1)
        show_image(image.detach().cpu(), epoch, i)
    
    return
#--------------Predict stuff-------------------------#
   
def training(): 
    training_loader = get_data_loaders(TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    model = UNETModel().to(DEVICE)
    if LOAD_MODEL:
        model.load_state_dict(torch.load("Model.pt"))

    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
        
    vloss = []
    
    #Training Loop
    for epoch in range(NUM_EPOCHS):
        for i, images in enumerate(tqdm(training_loader)):
            if i < 85:            
                optimizer.zero_grad()
            
                time = torch.randint(0, NOISE_STEP, (BATCH_SIZE,), device=DEVICE).long()
                loss = model_loss(model, images, time)
                vloss.append(loss.item())            
            
                loss.backward()
                optimizer.step()

                #Save the model every 10 iterations in each epoch
                if epoch % 1 == 0 and i % 10 == 0:
                    print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
                    torch.save(model.state_dict(), "Model.pt")
                    # plot_v = torch.tensor(vloss, device='cpu')
                    # plt.title('Stable Diffusion Loss')
                    # plt.plot(plot_v, color = 'blue')
                    # plt.savefig('loss_plot')
                if epoch % 1 == 0 and i == 0:
                    sample_image(epoch, i, model)

if __name__ == '__main__':
    training()