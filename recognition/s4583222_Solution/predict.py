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
from modules import UNETModel
import numpy as np
from PIL import Image
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 #If we drop this to 1, then len(training_loader) = 11,328
IMG_HEIGHT = 128 #Actual is 256
IMG_WIDTH = 128 #Actual is 256
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

def show_image(image, epoch, i, disc = "Blank"):
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
    plt.savefig(f"{epoch}_{i}_{disc}")

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
        noise = sample_image_at_timestep(noise, t, model)
        # if ind % stepsize == 0:
        #     plt.subplot(1, num_images, int(ind/stepsize)+1)
    show_image(noise.detach().cpu(), epoch, i, f"{ind}_backward")    

def predict():
    model = UNETModel().to(DEVICE)
    model.load_state_dict(torch.load("Model_2.pt"))
    gen_images = 1 #Generate 20 images
    images_generated = 0
    iteration = 0
    while images_generated < gen_images:
        sample_image(images_generated + 1, iteration, model)
        images_generated = images_generated + 1
    print("Done!")

if __name__ == '__main__':
    predict()