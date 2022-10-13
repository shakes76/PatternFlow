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
BATCH_SIZE = 128 #If we drop this to 1, then len(training_loader) = 11,328
IMG_HEIGHT = 32 #Actual is 256
IMG_WIDTH = 32 #Actual is 256
TRAINING_DIR = '/home/Student/s4583222/COMP3710/Images/Train'

def index_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1,t.cpu())
    out = out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)
    return out

@torch.no_grad()
def sample_image_at_timestep(rand_noise, t, model, beta, alpha_hat, sqrt_alpha_reciprical, prosterior_var):

    beta_time = index_list(beta, t, rand_noise.shape)
    sqrt_one_minus_alpha_hat_time = index_list(torch.sqrt(1. - alpha_hat), t, rand_noise.shape)
    sqrt_alpha_reciprical_time = index_list(sqrt_alpha_reciprical, t, rand_noise.shape)
    prosterior_var_time = index_list(prosterior_var, t, rand_noise.shape)

    predicted_noise = model(rand_noise, t)
    mean = sqrt_alpha_reciprical_time * (rand_noise - beta_time * predicted_noise / sqrt_one_minus_alpha_hat_time)
    if t == 0:
        return mean
    else:
        noise = torch.randn_like(rand_noise)
        return mean + torch.sqrt(prosterior_var_time) * noise  

@torch.no_grad()
def sample_image(num_images_to_gen, model):
    num = 69
    # for num in range(1,num_images_to_gen):
    noise = torch.randn((1,3, IMG_HEIGHT, IMG_WIDTH), device=DEVICE)   
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(1000/num_images)
    beta = torch.linspace(0.0001, 0.2, 1000)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0) #alpha hat is the cumulative product of alpha
    alpha_hat_prev = F.pad(alpha_hat[:-1], (1,0), value=1.0)
    sqrt_alpha_reciprical = torch.sqrt(1.0/alpha)
    prosterior_var = beta * (1. - alpha_hat_prev)/(1. - alpha_hat)
    for ind in range(1000, -1, 1):

        t = torch.full((1,), ind, device=DEVICE, dtype=torch.long)
        image = sample_image_at_timestep(noise, t, model, beta, alpha_hat, sqrt_alpha_reciprical, prosterior_var)
        if ind % stepsize == 0:
            plt.subplot(1, num_images, int(ind/stepsize)+1)
            image = image.detach().cpu()
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
    plt.savefig(f"image_{num}")
    return


def predict():
    model = UNETModel().to(DEVICE)
    model.load_state_dict(torch.load("Model_2.pt"))
    gen_images = 20 #Generate 20 images
    sample_image(gen_images, model)
    print("Done!")

if __name__ == '__main__':
    predict()
