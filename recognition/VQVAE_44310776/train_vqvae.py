"""
Training script for the VQVAE2 model. Supports Distibuted Data Parallel training on multi-GPU.
"""

import os
import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import transforms, utils as vutils
from vqvae import ResponsiveVQVAE2 as VQVAE2
from load_dataset import OASISDataset
from trainer import Trainer
from dotenv import load_dotenv

########################
# Constants
########################
RESULTS_DIRECTORY = "results"
NOTION_NAME = None
SAVE_NAME = "model.pth"
LATENT_DIMENSIONS = (32, 16)
DATASET_DATA = {
    'oasis': {
        'channels': 1,
        'dimension': 64 # Target dimension for the images.
    }
}
########################


########################
# Hyperparameters
########################
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99

learning_rate = 1e-3
########################

########################
# Model Definition
########################
def create_model(dataset, batch_size):
    in_channels = DATASET_DATA[dataset]['channels']
    dimension = DATASET_DATA[dataset]['dimension']

    model = VQVAE2(dimension, LATENT_DIMENSIONS, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, decay)

    # Dummy forward pass to initialise weights before distributing. Required for PyTorch DDP.
    model(torch.zeros(batch_size, in_channels, dimension, dimension))

    return model
########################

########################
# Dataset Setup
########################
def create_data_loaders(rank: int, world_size: int, batch_size: int, dataset: str, test_percentage: int) -> Tuple[DataLoader, DataLoader]:
    data_root = os.getenv("SLICES_PATH")

    # Image transforms.
    transform = transforms.Compose([
            transforms.Resize(DATASET_DATA[dataset]['dimension']),
            transforms.CenterCrop(DATASET_DATA[dataset]['dimension']),
            transforms.ToTensor(),
            # Don't need to normalize because it is done in __getitem__ (see load_dataset.py).
    ])

    # Load dataset and split into training and test sets.
    dataset = OASISDataset(data_root, transform=transform)
    test_size = len(dataset) // test_percentage
    main_data, _ = random_split(dataset, [len(dataset) - test_size, test_size], generator=torch.Generator().manual_seed(42))

    # Split training set into training and validation sets.
    validation_size = len(main_data) // test_percentage
    training_data, validation_data = random_split(main_data, [len(main_data) - validation_size, validation_size], generator=torch.Generator().manual_seed(42))

    # Create a sampler for distributed loading.
    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True, seed=42)

    # Make Dataloaders for each dataset.
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False, num_workers=16, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader
########################

def main(rank: int, epochs: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    # Put model onto the relevant device for this process rank.
    device = torch.device(f'cuda:{rank}')
    print(f'Rank {rank} using {torch.cuda.get_device_name(device)}.')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

########################
# Train Model
########################
    # Define optimiser and loss functions.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    mse_loss = nn.MSELoss()

    # Concrete subclass for Trainer.
    class VQVAE2Trainer(Trainer):
        def step(self, batch):
            data = batch.to(self._device)
            reconstruction, vq_loss = self._model(data)
            loss = {"MSE": mse_loss(reconstruction, data), "VQ": vq_loss.mean() * commitment_cost}
            return loss, 0

    # Instantiate the Trainer.
    trainer = VQVAE2Trainer(model, train_loader, val_loader, optimizer, device=device, ddp=True, rank=rank)

    # Define a callback to print the loss and accuracy each epoch.
    def plot():
        trainer.plot_loss(save_path=RESULTS_DIRECTORY, quiet=True, yscale='log')
        trainer.save_model(os.path.join(RESULTS_DIRECTORY, SAVE_NAME))

    # Define a callback to produce some sample reconstructions each epoch.
    def sample():
        model.eval()
        originals = next(iter(val_loader))
        originals = originals.to(device)
        reconstructions = None
        with torch.no_grad():
            reconstructions, _ = model(originals)

        # Plot the real images
        plt.figure(figsize=(10,7))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Original Images")
        plt.imshow(np.transpose(vutils.make_grid(originals, padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Reconstructed Images")
        plt.imshow(np.transpose(vutils.make_grid(reconstructions, padding=5, normalize=True).cpu(),(1,2,0)))
        plt.savefig(os.path.join(RESULTS_DIRECTORY, "reconstructions.png"))
        plt.close()

    # Register callbacks with the Trainer.
    trainer.set_callback(plot, np.linspace(1, epochs, epochs//5,  endpoint=False, dtype=int))
    trainer.set_callback(sample, np.linspace(1, epochs, epochs//5,  endpoint=False, dtype=int))

    # Train the model. Catch keyboard interrupts for clean exiting.
    try:
        with torch.autograd.set_detect_anomaly(True):
            trainer.train(epochs, quiet=False)
    except KeyboardInterrupt as e:
        print("User interrupted training, saving current results and terminating.")
        return trainer
    
    return trainer
########################


if __name__ == '__main__':
########################
# Args and Env Vars
########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help="Training batch size. Default=64.")
    parser.add_argument('--dataset', type=str, default='oasis', help="Which dataset to use. Default=oasis.")
    parser.add_argument('--test', type=int, default=20, help="Percentage of dataset to use for testing. Default=20.")
    parser.add_argument('--savename', type=str, default="model.pth", help='Name for saved model file. Default=model.pth.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for. Default=100.')
    args = parser.parse_args()
    load_dotenv()
########################
    # Set conditional constants.
    RESULTS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, args.dataset)
    SAVE_NAME = args.savename

    # Setup DDP things.
    rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv('WORLD_SIZE'))

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    # Setup data and train the model.
    train_loader, val_loader = create_data_loaders(rank, 
                                                   world_size, 
                                                   args.batch,
                                                   args.dataset,
                                                   args.test)

    trainer = main(rank=rank,
                 epochs=args.epochs,
                 model=create_model(args.dataset, args.batch),
                 train_loader=train_loader,
                 val_loader=val_loader)

########################
# Save Trainer
########################
    if rank == 0:
        try:
            os.mkdir(RESULTS_DIRECTORY)
        except FileExistsError as FEE:
            print(f"Directory '{RESULTS_DIRECTORY}' already exists.")
        trainer.plot_loss(save_path=RESULTS_DIRECTORY, quiet=True, yscale='log')
        trainer.save_model(os.path.join(RESULTS_DIRECTORY, SAVE_NAME))
########################