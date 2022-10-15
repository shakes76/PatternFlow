import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from modules import *
from dataset import *

torch.manual_seed(3710)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 20
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 16
LEARNING_RATE = 3e-4
NUM_CHANNELS = 3
D = 256
K = 512

def train(dataloader, model, optimiser):
    for real_images, _ in dataloader:
        real_images = real_images.to(device=DEVICE)

        optimiser.zero_grad()
        vq_loss, recon_image = model(real_images)

        recon_error = F.mse_loss(recon_image, real_images)
        loss = recon_error + vq_loss
        loss.backward()
        optimiser.step()

def fit():
    train_loader, test_loader = get_data(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    model = VQVA(NUM_CHANNELS, D, K).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, normalize=True)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(np.transpose(fixed_grid, (1, 2, 0)))
    plt.show()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimiser)

        reconstruction = generate_samples(fixed_images, model).cpu()
        print('SSIM:', ssim(fixed_images, reconstruction))
        grid = make_grid(reconstruction, nrow=8, normalize=True)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()
