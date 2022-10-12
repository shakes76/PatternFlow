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
from modules import Diffusion
from modules import UNETModel
import numpy as np
from PIL import Image
import torch.nn.functional as F

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10 #If we drop this to 1, then len(training_loader) = 11,328
NUM_EPOCHS = 100
NUM_WORKERS = 0
IMG_HEIGHT = 256 #Actual is 256
IMG_WIDTH = 256 #Actual is 256
PIN_MEMORY = False
LOAD_MODEL = False
TRAINING_DIR = '/home/Student/s4583222/COMP3710/Images/Train'

def index_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1,t.cpu())
    out = out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)
    return out

def forward_diffusion_to_image(image, t, alpha_hat, device =DEVICE):
    sqrt_alpha_hat = index_list(torch.sqrt(alpha_hat), t, image.shape)
    sqrt_one_minus_alpha_hat = index_list(torch.sqrt(1. - alpha_hat), t, image.shape)

    epsilon = torch.randn_like(image)
    # print(f"Epoch: {(sqrt_alpha_hat * image).shape[0]} | iteration: {(sqrt_one_minus_alpha_hat * epsilon).shape[0]}")
    x_1 = sqrt_alpha_hat * image
    x_2 = sqrt_one_minus_alpha_hat * epsilon
    if x_1.shape[0] != x_2.shape[0]:
        x_1 = SAFETY_STRAT_1
        x_2 = SAFETY_STRAT_2
    else:
        SAFETY_STRAT_1 = x_1
        SAFETY_STRAT_2 = x_2         
    x_t = x_1.to(device) + x_2.to(device)
    return x_t, epsilon.to(device)

def model_loss(model, image, t, alpha_hat):
    noisy_image, actual_noise = forward_diffusion_to_image(image, t, alpha_hat)
    predicted_noise = model(noisy_image, t)
    loss = F.l1_loss(actual_noise, predicted_noise)
    return loss
    
def training(): 
    training_loader = get_data_loaders(TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    beta = torch.linspace(0.0001, 0.2, 1000)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0) #alpha hat is the cumulative product of alpha
    # Can be used to show training image    
    # image = next(iter(training_loader))[0]
    # plt.figure(figsize=(15,15))
    # plt.axis('off')
    # num_images = 10
    # stepsize = int(100/num_images)
    # for ind in range(0, 100, stepsize):
    #     t = torch.Tensor([ind]).type(torch.int64)
    #     plt.subplot(1, num_images+1, int(ind/stepsize)+1)
    #     image, noise = forward_diffusion_to_image(image, t, alpha_hat)
    #     reverse_transforms = transforms.Compose([
    #         transforms.Lambda(lambda t: (t + 1) / 2),
    #         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #         transforms.Lambda(lambda t: t * 255.),
    #         transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    #         transforms.ToPILImage(),
    #     ])
    #     # Take first image of batch
    #     if len(image.shape) == 4:
    #         image = image[0, :, :, :] 
    #     plt.imshow(reverse_transforms(image))
    #     plt.savefig("test")

    model = UNETModel().to(DEVICE)
    if LOAD_MODEL:
        model.load_state_dict(torch.load("Model_2.pt"))

    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
        
    vloss = []
    
    #Training Loop
    for epoch in range(NUM_EPOCHS):
        for i, images in enumerate(tqdm(training_loader)):
            if i < 1131:            
                optimizer.zero_grad()
            
                time = torch.randint(0, 1000, (BATCH_SIZE,)).long()
                loss = model_loss(model, images, time, alpha_hat)
                vloss.append(loss.item())            
            
                loss.backward()
                optimizer.step()

                #Save the model every epoch
                if epoch % 1 == 0 and i % 10 == 0:
                    print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
                    torch.save(model.state_dict(), "Model_2.pt")
                    plot_v = torch.tensor(vloss, device='cpu')
                    plt.title('Stable Diffusion Loss')
                    plt.plot(plot_v, color = 'blue')
                    plt.savefig('loss_plot')

                if epoch % 10 == 0 and i == 0:
                    print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
                    torch.save(model.state_dict(), "Model_2.pt")
                    #Display image

if __name__ == '__main__':
    training()