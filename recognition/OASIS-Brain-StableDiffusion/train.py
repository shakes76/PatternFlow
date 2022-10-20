import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from modules import *
from dataset import *
from utils import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torch.optim import Adam

def train(device, lr, path, model, epochs, batch_size):
    dataloader = load_dataset(path, batch_size=batch_size)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    length = len(dataloader)

    for epoch in range(epochs):
        for idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            pos = get_sample_pos(data.shape[0]).to(device)
            noisy_x, noise = add_noise(data, pos)
            predicted_noise = model(noisy_x, pos)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "./models/")

def main():
    #hyperparameters
    device = "cuda"
    lr = 3e-4
    path = r"D:\COMP3710\Stable Diffusion\code\OASIS-Brain-Data\training_data"
    model = UNet().to(device)
    batch_size = 2
    epochs = 200

    # start training
    train(device, lr, path, model, epochs, batch_size)

if __name__ == "__main__":
    main()