import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from modules import *
from dataset import *
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import pickle

# Seed the random number generator for reproducibility of the results
torch.manual_seed(3710)

SSIM = StructuralSimilarityIndexMeasure()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 20
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 16
LEARNING_RATE = 3e-4
NUM_CHANNELS = 3
D = 256
K = 512

def train(dataloader, model, optimiser):
    recon_losses = 0.0
    for real_images, _ in dataloader:
        real_images = real_images.to(DEVICE)

        optimiser.zero_grad()
        vq_loss, recon_image = model(real_images)

        recon_error = F.mse_loss(recon_image, real_images)
        recon_losses += recon_error
        loss = recon_error + vq_loss
        loss.backward()
        optimiser.step()
    return (recon_losses / len(dataloader)).item()

def fit():
    train_loader, test_loader = get_data(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    model = VQVAE(NUM_CHANNELS, D, K).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    fixed_images, _ = next(iter(test_loader))
    save_image(fixed_images, fp='Test.png', nrow=8, normalize=True)

    best_ssim = 0
    recon_losses = []
    for epoch in range(NUM_EPOCHS):
        loss = train(train_loader, model, optimiser)

        with torch.no_grad():
            _, recontructed = model(fixed_images.to(DEVICE))

        recontructed = recontructed.cpu()
        ssim = SSIM(fixed_images, recontructed)
        print(f'Epoch {epoch}: Reconstruction Loss - {loss}, SSIM - {ssim}')
        if ssim > best_ssim:
            save_image(recontructed, fp='Reconstructed.png', nrow=8, normalize=True)
            with open('VQVAE', 'wb') as f:
                pickle.dump(model, f)

        recon_losses.append(loss)

    plt.plot(range(1, NUM_EPOCHS + 1), recon_losses)
    plt.savefig('Loss.png')

fit()