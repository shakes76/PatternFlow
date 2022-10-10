# COMP3710 Pattern Recognition Lab Assignment
# By Thomas Jellett (s4583222)
# HARD DIFFICULTY
# Create a generative model of the OASIS brain using stable diffusion that
# has a “reasonably clear image.”

# File: train.py
# Description: Used to train the model

import os
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from dataset import get_data_loaders
from modules import Diffusion
from modules import UNETModel
import numpy as np
from PIL import Image

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 30 #If we drop this to 1, then len(training_loader) = 11,328
NUM_EPOCHS = 1000
NUM_WORKERS = 0
IMG_HEIGHT = 256 #Actual is 256
IMG_WIDTH = 256 #Actual is 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAINING_DIR = 's4583222_Solution/Images/Train'

def training(): 
    training_loader = get_data_loaders(TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    model = UNETModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    mse = nn.MSELoss()
    destroy_image = Diffusion(img_size = IMG_HEIGHT)
    length = len(training_loader)

    #Training Loop
    for epoch in range(NUM_EPOCHS):
        progress = tqdm(training_loader)
        for i, images in enumerate(training_loader):
            images = images.to(DEVICE)
            time = destroy_image.sample_the_timestep(images.shape[0]).to(DEVICE)
            x_t, actual_noise = destroy_image.forward_process(images, time)
            predicted_noise = model(x_t, time)
            loss = mse(actual_noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix(mse = loss.item())
            if epoch % 1 == 0 and i % 10 == 0:
                print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
                # torch.save(model.state_dict(), os.path.join("models", "STABLEDIFFUSION", f"ckpt.pt"))
                torch.save(model.state_dict(), "Model.pt")
            if epoch % 100 == 99 and i == 0:
                print(f"Epoch: {epoch} | iteration: {i} | Loss: {loss.item()}")
                # torch.save(model.state_dict(), os.path.join("models", "STABLEDIFFUSION", f"ckpt.pt"))
                torch.save(model.state_dict(), "Model.pt")
                sampled = destroy_image.sample(model, n=images.shape[0])
                
                plt.figure(figsize = (256,256))
                plt.imshow(torch.cat([torch.cat([j for j in sampled.cpu()], dim = -1)], dim = -2).permute(1,2,0).to('cpu'))
                plt.show()

                grid = torchvision.utils.make_grid(images)
                ndarr = grid.permute(1,2,0).to('cpu').numpy()
                im = Image.fromarray(ndarr)
                im.save(f"{epoch}.jpg")





            



if __name__ == '__main__':
    training()