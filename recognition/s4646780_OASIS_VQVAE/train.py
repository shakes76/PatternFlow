__author__ = "Utkarsh Sharma"

import numpy as np
import torch
import torch.nn.functional as F
from dataset import create_train_test_loaders_vqvae, create_train_test_loaders_dcgan
from modules import VQVAE
from models.generator import Generator
from models.discriminator import Discriminator
from utils import initialise_weights
import torch.nn as nn
from constants import DEVICE, ROOT_DIR


def train_VQVAE(num_epochs, lr):
    """
    Function to train the VQ-VAE model with the OASIS dataset.
    """
    train_loader, test_loader = create_train_test_loaders_vqvae(ROOT_DIR)
    NUM_EMBEDDINGS = 512  # number of embeddings for codebook
    EMBEDDING_DIM = 64  # embedding dimension
    BETA_COST = 0.25  # commitment cost for VQ
    DATA_VARIANCE = 0.0338  # evaluated separately on training data
    train_res_recon_error = []

    model = VQVAE(NUM_EMBEDDINGS, EMBEDDING_DIM, BETA_COST).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()  # sets the model in training model so gradients are tracked

    for i in range(num_epochs):
        data = next(iter(train_loader))  # gets the next training data
        data = data.to(DEVICE)
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon,
                                 data) / DATA_VARIANCE  # reconstructed loss with is an MSE loss scaled by var
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        train_res_recon_error.append(recon_error.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print()

    return model, train_res_recon_error, test_loader


def train_DCGAN(vqvae, gan_epochs, lr_gan, batch_size):
    """
    Function to train the DCGAN model using the encoder and VQ to create the dataset.
    """
    train_loader_gan, test_loader_gan = create_train_test_loaders_dcgan(vqvae, batch_size, ROOT_DIR)

    generator = Generator()
    discriminator = Discriminator()

    # initialize weights for DCGAN
    initialise_weights(generator)
    initialise_weights(discriminator)

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=lr_gan, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=lr_gan, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    # Training the DCGAN
    for epoch in range(gan_epochs):
        for batch, real_image in enumerate(train_loader_gan):
            real_image = real_image.to(DEVICE)
            noise = torch.randn(batch_size, 100, 1, 1).to(DEVICE)
            fake_image = generator(noise)

            # Train discriminator
            disc_real = discriminator(real_image).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
            disc_fake = discriminator(fake_image.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            discriminator.zero_grad()
            loss_disc.backward()
            optimizer_dis.step()

            # train generator
            output = discriminator(fake_image).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            if batch % 100 == 0:
                print(
                    f"Epoch [{epoch}/{gan_epochs}] Batch {batch}/{len(train_loader_gan)} \
                                  Loss D: {loss_disc:.3f}, loss G: {loss_gen:.3f}")

    return generator, torch.unique(real_image)
