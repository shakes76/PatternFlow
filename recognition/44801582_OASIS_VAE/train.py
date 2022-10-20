"""
Source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training.
"""
from dataset import *

import torch
from tensorboardX import SummaryWriter
from datetime import datetime

def main(device):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    writer = SummaryWriter(f"./logs/{current_time}")
    save_filename = f"./models/{current_time}"

    train_loader, test_loader, validation_loader = get_loaders()


if __name__ == "__main__":
    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Device
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else 'cpu')

    main(device)
