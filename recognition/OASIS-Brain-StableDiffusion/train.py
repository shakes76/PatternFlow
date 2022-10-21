import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.optim import Adam
import csv
from modules import *
from dataset import *
from utils import *

def train(device, lr, train_path, test_path, model, epochs, batch_size):
    """
    Training loop to train the model

    Args:
        device (string): device to train on 
        lr (float): learning rate of the model
        train_path (string): path to training data
        test_path (string): path to test data
        model (Module): network model
        epochs (int): number of epochs to run
        batch_size (int): batch size
    """
    train_dataloader = load_dataset(train_path, batch_size=batch_size)
    dataloader_length = len(train_dataloader)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tracked_loss = []
    test_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for idx, (data, _) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            position = get_sample_pos(data.shape[0]).to(device)
            noisy_x, noise = add_noise(data, position)
            predicted_noise = model(noisy_x, position)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        
        tracked_loss.append([epoch_loss / dataloader_length])
        print("Current Loss ==> {}".format(epoch_loss/dataloader_length))
        test_loss.append(test_model(model, test_path, batch_size, device))

    torch.save(model.state_dict(), "./model/")
    save_loss_data(tracked_loss, test_loss)

def test_model(model, test_path, batch_size, device):
    """
    Test loop to test the model against test data

    Args:
        model (Module): model to test
        test_path (string): path to the test data
        batch_size (int): batch size
        device (string): device to use

    Returns:
        List: index of loss of the model against the test data
    """
    test_dataloader = load_dataset(test_path, batch_size=batch_size)
    dataloader_length = len(test_dataloader)

    model = model.to(device)

    loss_fn = nn.MSELoss()
    running_loss = 0
    for idx, (data, _) in enumerate(tqdm(test_dataloader)):
        data = data.to(device)
        position = get_sample_pos(data.shape[0]).to(device)
        noisy_x, noise = add_noise(data, position)
        predicted_noise = model(noisy_x, position)
        loss = loss_fn(noise, predicted_noise)
        running_loss += loss.item()

    print("Test Loss ==> {}".format(running_loss / dataloader_length))
    return [running_loss / dataloader_length]

def main():
    #hyperparameters
    device = "cuda"
    lr = 3e-4
    train_path = r".\OASIS-Brain-Data\training_data"
    test_path = r".\OASIS-Brain-Data\test_data"
    model = UNet().to(device)
    batch_size = 2
    epochs = 200

    # start training
    train(device, lr, train_path, test_path, model, epochs, batch_size)

if __name__ == "__main__":
    main()