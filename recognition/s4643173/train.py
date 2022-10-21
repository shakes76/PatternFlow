import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from modules import *
from dataset import *
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import pickle
import torch.nn as nn

SSIM = StructuralSimilarityIndexMeasure()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 20
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 16
LEARNING_RATE = 3e-4
NUM_CHANNELS = 3
D = 256
K = 512
NUM_EPOCHS_GAN = 40
LEARNING_RATE_GAN = 2e-4


def train(dataloader, model, optimiser):
    """ 
    Training loop for the VQ-VAE model. 
    
    Parameters:
        dataloader: the dataloader for the train set.
        model: the VQ-VAE model. 
        optimiser: the optimiser.

    Returns:
        The average reconstruction loss for the epoch.
    """
    recon_losses = 0.0
    for real_images, _ in dataloader:
        real_images = real_images.to(DEVICE)

        optimiser.zero_grad()
        # Forward pass through the model
        vq_loss, recon_image = model(real_images)

        recon_error = F.mse_loss(recon_image, real_images)
        recon_losses += recon_error
        loss = recon_error + vq_loss
        # Calculate the gradients for the model through a backward pass
        loss.backward()
        optimiser.step()
    return (recon_losses / len(dataloader)).item()


def fit_vqvae():
    """ 
    Creates and trains the VQ-VAE model, saves the model with the highest
    SSIM and saves and returns the highest SSIM achieved by the model. 
    """
    train_loader, test_loader = get_data(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    model = VQVAE(NUM_CHANNELS, D, K).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fixed_images, _ = next(iter(test_loader))
    save_image(fixed_images, fp='Test.png', normalize=True)

    best_ssim = 0
    recon_losses = []
    for epoch in range(NUM_EPOCHS):
        loss = train(train_loader, model, optimiser)

        with torch.no_grad():
            _, recontructed = model(fixed_images.to(DEVICE))

        recontructed = recontructed.cpu()
        # Calculate the SSIM between the test images and the reconstructed 
        # images.
        ssim = SSIM(fixed_images, recontructed)
        print(f'Epoch {epoch + 1}: Reconstruction Loss - {loss}, SSIM - {ssim}')
        if ssim > best_ssim:
            save_image(recontructed, fp='Reconstructed.png', normalize=True)
            best_ssim = ssim
            with open('VQVAE', 'wb') as f:
                pickle.dump(model, f)

        recon_losses.append(loss)

    plt.plot(range(1, NUM_EPOCHS + 1), recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(2, NUM_EPOCHS + 1, 2), list(range(2, NUM_EPOCHS + 1, 2)))
    plt.savefig('Loss.png')

    return best_ssim


def fit_gan():
    """ Loads the VQVAE model, creates, trains and saves the DCGAN model. """
    try:
        with open('./VQVAE', 'rb') as f:
            vae_model = pickle.load(f)
    except FileNotFoundError:
        print('VQVAE model not found!')
        exit(1)

    gen = Generator()
    disc = Discriminator()
    gen = gen.to(DEVICE)
    disc = disc.to(DEVICE)

    optimiser_G = torch.optim.Adam(
        gen.parameters(), lr=LEARNING_RATE_GAN, betas=(0.5, 0.999)
    )
    optimiser_D = torch.optim.Adam(
        disc.parameters(), lr=LEARNING_RATE_GAN, betas=(0.5, 0.999)
    )
    criterion = nn.BCELoss()


    dataloader, _ = get_data(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    torch.cuda.empty_cache()
    avg_gen_loss = []
    avg_disc_loss = []
    for epoch in range(NUM_EPOCHS_GAN):
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        for batch, (image, _) in enumerate(dataloader):
            image = image.to(DEVICE)
            _, _, _, indice = vae_model.codebook(vae_model.encoder(image))
            indice = indice.float().to(DEVICE)
            indice = indice.view(BATCH_SIZE_TRAIN, 64, 64)
            # Convert to 3 channels
            indice = torch.stack((indice,) * 3, 0)
            # Reshape to BCHW
            indice = indice.permute(1, 0, 2, 3)

            # Train the discriminator to maximise log(D(x)) + log(1 - D(G(z)))
            optimiser_D.zero_grad()
            b_size = indice.size(0)

            pred = disc(indice).view(-1)
            target = torch.ones_like(pred)
            loss = criterion(pred, target)

            noise = torch.randn(b_size, 100, 1, 1, device=DEVICE)
            # Generate fake codebook indices
            fake_data = gen(noise)

            # Classify the fake indices
            fake_pred = disc(fake_data).view(-1)
            fake_target = torch.zeros_like(fake_pred)
            fake_loss = criterion(fake_pred, fake_target)

            # Compute the loss of D as sum of the fake and the real indices
            dis_loss = loss + fake_loss
            total_disc_loss += dis_loss.item()
            dis_loss.backward()
            optimiser_D.step()

            # Train the generator to maximise log(D(G(z)))
            optimiser_G.zero_grad()
            noise = torch.randn(b_size, 100, 1, 1, device=DEVICE)
            fake_data = gen(noise)
            preds = disc(fake_data).view(-1)
            target = torch.ones_like(preds)
            gen_loss = criterion(preds, target)

            total_gen_loss += gen_loss.item()
            gen_loss.backward()
            optimiser_G.step()

            if batch != 0 and batch % 100 == 0:
                print(
                    f'Epoch {epoch + 1}: Gen Loss - {gen_loss.item():.4f}' + \
                        f', Disc Loss - {dis_loss.item():.4f}'
                )

        avg_gen_loss.append((total_gen_loss / len(dataloader)))
        avg_disc_loss.append((total_disc_loss / len(dataloader)))

    with open('Gen', 'wb') as f:
        pickle.dump(gen, f)
    plt.plot(range(1, NUM_EPOCHS_GAN + 1), avg_disc_loss)
    plt.plot(range(1, NUM_EPOCHS_GAN + 1), avg_gen_loss)
    plt.xlim(1, NUM_EPOCHS_GAN)
    plt.title('DCGAN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.savefig('DCGAN Loss.png')
