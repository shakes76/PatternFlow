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
LOAD_MODEL = False #Set to True when you want to load a pre-trained model
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

def forward_diffusion_to_image(image, t, device =DEVICE):
    """
    This is the forward diffusion process. It will noise an image to a timestep and then return that image as well as the actual
    noise of the image.
    Acknowledgment From:   
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=592aa765
    """
    sqrt_alpha_hat = index_list(torch.sqrt(ALPHA_HAT), t, image.shape)
    sqrt_one_minus_alpha_hat = index_list(torch.sqrt(1. - ALPHA_HAT), t, image.shape)
    epsilon = torch.randn_like(image)      
    return sqrt_alpha_hat.to(device) * image.to(device) + sqrt_one_minus_alpha_hat.to(device) * epsilon.to(device), epsilon.to(device)
   
def training():
    """
    The main training loop.
    As well as saving the model and plotting the loss.
    """ 
    training_loader = get_data_loaders(TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)#Get the data loader for the training images
    model = UNETModel().to(DEVICE)#Define the model
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)#Optimizer 
    if LOAD_MODEL: #If True, load a pretrained model
        model.load_state_dict(torch.load("Model_2.pt"))

    vloss = []#List to fill with all the loss values
    
    #Training Loop
    for epoch in range(NUM_EPOCHS):
        for i, images in enumerate(tqdm(training_loader)):          
            optimizer.zero_grad()
        
            time = torch.randint(0, NOISE_STEP, (BATCH_SIZE,), device=DEVICE).long() #Random timestep to noise the image
            noisy_image, actual_noise = forward_diffusion_to_image(images, time)#Noise the image
            predicted_noise = model(noisy_image, time)#Given the noisy image and to the model and let it predict how much noise has been added
            loss = F.l1_loss(actual_noise, predicted_noise)#Calculate the mean absolute error between the actual noise and the predicted noise 
            vloss.append(loss.item())#Append loss to list            
        
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0 and i % 10 == 0:#Save the model every 10 iterations in each epoch
                torch.save(model.state_dict(), "Model_2.pt")#Save the weights of the model
                plot_v = torch.tensor(vloss, device='cpu')#Turn the vloss list into a tensor and give it to the cpu
                plt.title('Stable Diffusion Loss')
                plt.plot(plot_v, color = 'blue')#Plot the loss curve
                plt.savefig('loss_plot')#Save the loss plot

            if epoch % 5 == 0 and i == 1: #Creates back up model in case other model saves incorrectly
                print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")#Prints to .out file
                torch.save(model.state_dict(), "Model_backup.pt")
                
if __name__ == '__main__':
    training()