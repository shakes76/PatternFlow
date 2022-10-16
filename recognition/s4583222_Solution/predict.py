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

NOISE_STEP = 600 #How many timesteps we want to destroy an image

BETA = torch.linspace(0.0001, 0.02, NOISE_STEP) #Linear beta schedule
ALPHA = 1. - BETA
ALPHA_HAT = torch.cumprod(ALPHA, axis=0) #alpha hat is the cumulative product of alpha
ALPHA_HAT_PREV = F.pad(ALPHA_HAT[:-1], (1,0), value=1.0) #Previous alpha hat
SQRT_ALPHA_REC = torch.sqrt(1.0/ALPHA) #Square root of the reciprical of alpha
PROS_VAR = BETA * (1. - ALPHA_HAT_PREV)/(1. - ALPHA_HAT) #Calculation for the posterior

def index_list(vals, t, x_shape):
    """
    Returns the time index for a batch
    Acknowledgment From:   
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=592aa765
    """
    batch_size = t.shape[0]
    out = vals.gather(-1,t.cpu())
    out = out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)
    return out

def show_image(image, epoch, i, disc = "Blank"):
    """
    Takes in an image tensor and converts it to an image. The image will be coloured so it will save it and open the image again to
    convert it to a grayscale image. This image is them saved
    Acknowledgment From:   
    https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=Rj17psVw7Shg
    """    
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2), #Renormalise tensor
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), #Move chanels to the back
        transforms.Lambda(lambda t: t * 255.),#Get pixels back to colour
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),#Convert to numpy array
        transforms.ToPILImage(),#Convert to image
    ])
    
    if len(image.shape) == 4:# Take first image of batch
        image = image[0, :, :, :] 

    plt.imshow(reverse_transforms(image))
    plt.savefig(f"{epoch}_{i}_{disc}_colour") #Saves coloured brain image
    im = Image.open(f"{epoch}_{i}_{disc}_colour.png").convert("L") #Opens coloured brain image as gray scale
    plt.imshow(im, cmap='gray')
    plt.savefig(f"{epoch}_{i}_{disc}_gray")#Saves grayscale image




@torch.no_grad() #Required so the weights of previous images does not effect the image we are generating
def sample_image_at_timestep(rand_noise, t, model):
    """
    Returns the time index for a batch
    Acknowledgment From:   
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=592aa765
    """    
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

@torch.no_grad() #Required so the weights of previous images does not effect the image we are generating
def sample_image(epoch, i, model):
    """
    This function is looped through and it loops through timesteps, slowly denoising an image of random noise.
    Acknowledgment From:   
    https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=Rj17psVw7Shg
    """
    noise = torch.randn((1,3, IMG_HEIGHT, IMG_WIDTH), device=DEVICE)#Random noise   
    plt.figure(figsize=(15,15))
    plt.axis('off')
    for ind in range(0, NOISE_STEP)[::-1]:#Reversed from 600 to 0
        t = torch.full((1,), ind, device=DEVICE, dtype=torch.long)
        noise = sample_image_at_timestep(noise, t, model)#Denoised image
    show_image(noise.detach().cpu(), epoch, i, f"{ind}_backward")#After looped through all timesteps show image    

def predict():
    """
    This function loads a previously saved model and then loops a function gen_images number of times.
    It will product gen_images number of images.
    Prints Done! to .out when finished
    """
    model = UNETModel().to(DEVICE)
    model.load_state_dict(torch.load("Model_2.pt"))
    gen_images = 1 #Generate x number images
    images_generated = 0
    iteration = 0
    while images_generated < gen_images:
        sample_image(images_generated + 1, iteration, model)
        images_generated = images_generated + 1
    print("Done!")

if __name__ == '__main__':
    predict()
