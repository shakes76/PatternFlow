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
# from pixel_cnn import PixelCNN
from pixel_cnn.prior_models import TopPrior, BottomPrior
from load_dataset import OASISDataset
from pytorchtrainer.trainer import Trainer
from notiontoolkit.tqdm_notion import tqdm_notion
from dotenv import load_dotenv
from torchinfo import summary

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
        'dimension': 64
    }
}
########################


########################
# Hyperparameters
########################
epochs = 100

# VQVAE
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99

learning_rate = 1e-4
########################

########################
# VQVAE Definition
########################
def load_vqvae(dataset: str, batch_size: int, filename: str):
    in_channels = DATASET_DATA[dataset]['channels']
    dimension = DATASET_DATA[dataset]['dimension']

    model = VQVAE2(dimension, LATENT_DIMENSIONS, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim, decay)

    # Dummy forward pass to initialise weights before distributing.
    model(torch.zeros(batch_size, in_channels, dimension, dimension))

    state_dict = torch.load(os.path.join(RESULTS_DIRECTORY, filename))
    model.load_state_dict(state_dict)

    # Disable grad for model.
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model
########################

########################
# PixelCNN Definition
########################
# def load_top(batch_size: int):
#     # in_channels = DATASET_DATA[dataset]['channels']
#     # dimension = DATASET_DATA[dataset]['dimension']
#     latent_dim = LATENT_DIMENSIONS[1]

#     model = PixelCNN(codebook_size=num_embeddings, feature_channels=256, n_layers=15, conditional=False)

#     # Dummy forward pass to initialise weights before distributing.
#     model(torch.zeros(batch_size, latent_dim, latent_dim, dtype=int))

#     state_dict = torch.load(os.path.join(RESULTS_DIRECTORY, "generator_top.pth"))
#     model.load_state_dict(state_dict)

#     # Disable grad for model.
#     model.eval()
#     for p in model.parameters():
#         p.requires_grad = False

#     return model
########################


########################
# PixelCNN Definition
########################
def create_model(batch_size: int, level: str):
    # Dummy forward pass to initialise weights before distributing.
    if level == 'top':
        model = TopPrior()
        model(torch.zeros(batch_size, LATENT_DIMENSIONS[1], LATENT_DIMENSIONS[1], dtype=int))
    else:
        model = BottomPrior()
        model(torch.zeros(batch_size, LATENT_DIMENSIONS[0], LATENT_DIMENSIONS[0], dtype=int),
              torch.zeros(batch_size, LATENT_DIMENSIONS[1], LATENT_DIMENSIONS[1], dtype=int))

    return model
########################


########################
# Dataset Setup
########################
def create_data_loaders(rank: int, world_size: int, batch_size: int, dataset: str, test_percentage: int) -> Tuple[DataLoader, DataLoader]:
    data_root = os.getenv("SLICES_PATH")

    transform = transforms.Compose([
            transforms.Resize(DATASET_DATA[dataset]['dimension']),
            transforms.CenterCrop(DATASET_DATA[dataset]['dimension']),
            transforms.ToTensor(),
            # Don't need to normalize because it is done in __getitem__.
    ])

    dataset = OASISDataset(data_root, transform=transform)
    test_size = len(dataset) // test_percentage
    main_data, _ = random_split(dataset, [len(dataset) - test_size, test_size], generator=torch.Generator().manual_seed(42))
    validation_size = len(main_data) // test_percentage
    training_data, validation_data = random_split(main_data, [len(main_data) - validation_size, validation_size], generator=torch.Generator().manual_seed(42))

    # Create a sampler for distributed loading.
    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True, seed=42)

    # Make Dataloaders for each dataset.
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False, num_workers=16, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader
########################

def main(rank: int, epochs: int, vqvae: VQVAE2, model: nn.Module, level: str, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    # Put model onto the relevant device for this process rank.
    device = torch.device(f'cuda:{rank}')
    print(f'Rank {rank} using {torch.cuda.get_device_name(device)}.')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    vqvae = vqvae.to(device)

########################
# Train Model
########################
    # Define optimiser and loss functions.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    ce_loss = nn.CrossEntropyLoss()

    # Create loss and prediction functions for the training module.
    def loss_fn(output, target):
        pass
    def pred_fn(output):
        pass

    class PixelCNNTrainer(Trainer):
        def step(self, batch):
            data = batch.to(self._device)
            _, _, _, id_2, id_1 = vqvae.encode(data)

            logits = None
            if level == 'top':
                logits = self._model(id_2)
            else:
                logits = self._model(id_1, id_2)

            logits = logits.permute(0, 2, 3, 1).contiguous()
            logits = logits.view(-1, logits.shape[-1])
            loss = {
                "CE": ce_loss(logits, id_2.view(-1) if level == 'top' else id_1.view(-1))
            }
            return loss, 0

    # Instantiate the trainer.
    trainer = PixelCNNTrainer(model, train_loader, val_loader, optimizer, loss_fn, pred_fn, device=device, ddp=True, rank=rank)
    
    # Enable Notion integration to track training progress.
    if NOTION_NAME is not None:
        print("Notion integration enabled.")
        trainer.tqdm = tqdm_notion
        trainer.tqdm_kwargs = {"page_title": NOTION_NAME}

    # Define a callback to print the loss and accuracy each epoch.
    def plot():
        trainer.plot_loss(save_path=RESULTS_DIRECTORY, quiet=True, yscale='log')
        trainer.plot_accuracy(save_path=RESULTS_DIRECTORY, quiet=True)
        trainer.save_model(os.path.join(RESULTS_DIRECTORY, SAVE_NAME))

    def view_reconstructions():
        batch = next(iter(trainer._val_loader))
        batch = batch[:8] # Just do 8 samples for speed.
        data = batch.to(device)

        _, _, _, id_2, id_1 = vqvae.encode(data)

        x = id_2 if level == 'top' else id_1
        condition = None if level == 'top' else id_2
        logits = trainer._model(x, condition)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        # Get the most likely index from each pixel prediction.
        codes = torch.argmax(logits, dim=3, keepdim=False)
        # print(generated_codes.shape, generated_codes)

        images = vqvae.decode_codebook(codes if level == 'bottom' else id_1,
                              codes if level == 'top' else id_2)

        # Plot the samples
        plt.figure(figsize=(30,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Originals")
        plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, normalize=True).cpu(),(1,2,0)))
        
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("PixelCNN Reconstructions")
        plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig(os.path.join(RESULTS_DIRECTORY, "prior_reconstructions.png"))
        plt.close()

    
    trainer.set_callback(plot, np.linspace(1, epochs, epochs//5,  endpoint=False, dtype=int))
    # trainer.set_callback(view_reconstructions, np.linspace(1, epochs, epochs//5,  endpoint=False, dtype=int))

    if level == 'bottom':
        # pixelCNN_top = load_top(batch_size=4)
        # pixelCNN_top.to(device)

        def generate_samples():
            print("Generating Samples...")
            batch = next(iter(trainer._val_loader))
            batch = batch[:4] # Just do 4 samples for speed.
            data = batch.to(device)

            with torch.no_grad():
                _, _, _, id_2, _ = vqvae.encode(data)
                generated_codes = trainer._model.module.generate((32,32), 4, id_2)

            images = vqvae.decode_codebook(generated_codes, id_2)

            # Plot the real images
            plt.figure(figsize=(10,7))
            plt.axis("off")
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(images, padding=5, normalize=True).cpu(),(1,2,0)))
            plt.savefig(os.path.join(RESULTS_DIRECTORY, "prior_samples.png"))
            plt.close()

        # trainer.set_callback(generate_samples, np.linspace(1, epochs, epochs//5,  endpoint=False, dtype=int))

    # Train the model. Catch keyboard interrupts for clean exiting.
    try:
        with torch.autograd.set_detect_anomaly(True):
            trainer.train(epochs, quiet=args.quiet)
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
    parser.add_argument('--level', type=str, default='top', help="Which level PixelCNN to train. Default=top.")
    parser.add_argument('--test', type=str, default=20, help="Percentage of dataset to use for testing. Default=20.")
    parser.add_argument('--quiet', default=False, action="store_true", help="Surpress script output for headless environments. Default=False.")
    parser.add_argument('--notion', type=str, help='Name for Notion entry. Will not use Notion if not set.')
    parser.add_argument('--savename', type=str, default="model.pth", help='Name for saved model file. Default=model.pth.')
    parser.add_argument('--vqvae', type=str, default="model.pth", help='Name for pretrained VQVAE model file. Default=model.pth.')
    args = parser.parse_args()
    load_dotenv()
########################
    # Set conditional constants.
    RESULTS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, args.dataset)
    NOTION_NAME = args.notion
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
                 epochs=epochs,
                 vqvae=load_vqvae(args.dataset, args.batch, args.vqvae),
                 model=create_model(args.batch, args.level),
                 level=args.level,
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
        trainer.plot_loss(save_path=RESULTS_DIRECTORY, quiet=args.quiet, yscale='log')
        trainer.plot_accuracy(save_path=RESULTS_DIRECTORY, quiet=args.quiet)
        trainer.save_model(os.path.join(RESULTS_DIRECTORY, SAVE_NAME))
########################