"""
Source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training.
"""
from dataset import *
from modules import VQ_VAE

import torch
from tensorboardX import SummaryWriter
from datetime import datetime


def train(model, data, optimizer, writer):
    pass


def test(model, data, writer):
    pass


def run_epoch(model, train_data, test_data, optimizer, writer):
    train(model, train_data, optimizer, writer)
    test(model, test_data, writer)


def main():
    writer = SummaryWriter(f"./logs/{current_time}")

    train_loader, test_loader, validation_loader = get_loaders()

    model = VQ_VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(3):
        run_epoch(model, train_loader, test_loader, optimizer, writer)

        with open(f"{save_filename}/model_{epoch + 1}.pt", 'wb') as f:
            torch.save(model.state_dict(), f)


if __name__ == "__main__":
    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else 'cpu')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    save_filename = f"./models/{current_time}"
    os.makedirs(save_filenameBu)

    main()
