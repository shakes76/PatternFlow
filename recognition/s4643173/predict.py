from dataset import *
from modules import *
from train import *

import os
import pickle
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure

SSIM = StructuralSimilarityIndexMeasure()
BATCH_SIZE_TEST = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if 'VQVAE' not in os.listdir('.'):
        print('No saved VQVAE model found. Creating and training a new model.')
        print('-' * 50)
        ssim = fit_vqvae()
        print('The VQ-VAE model achieve a SSIM of:', ssim)
    if 'Gen' not in os.listdir('.'):
        print('No saved DCGAN model found. Creating and training a new model.')
        print('-' * 50)
        fit_gan()

    with open('VQVAE', 'rb') as f:
        vae_model = pickle.load(f)
    with open('Gen', 'rb') as f:
        gen = pickle.load(f)

    train_loader, test_loader = get_data(BATCH_SIZE_TEST, BATCH_SIZE_TEST)
    fixed_images, _ = next(iter(test_loader))
    _, _, _, indice = vae_model.codebook(vae_model.encoder(fixed_images.to(DEVICE)))
    indice = indice.view(BATCH_SIZE_TEST, 64, 64)
    indice = indice.unsqueeze(1).float()
    save_image(indice, fp='Test Indices.png', normalize=True)

    noise = torch.randn(BATCH_SIZE_TEST, 100, 1, 1).to(DEVICE)
    fake_indices = gen(noise)
    fake_indices = torch.flatten(transforms.functional.rgb_to_grayscale(fake_indices).squeeze(), 1)
    save_image(fake_indices.view(BATCH_SIZE_TEST, 64, 64).unsqueeze(1), fp='Fake Indices.png', normalize=True)

    for i in range(BATCH_SIZE_TEST):
        unique_vals = [98, 127, 260, 352, 419, 420]
        interval_size = (torch.max(fake_indices[i]) - torch.min(fake_indices[i])) / len(unique_vals)
        for j in range(len(unique_vals)):
            min = torch.min(fake_indices[i]) + j * interval_size
            fake_indices[i][torch.logical_and(min <= fake_indices[i], fake_indices[i] <= (min + interval_size))] = unique_vals[j]

    quantised = vae_model.codebook.quantise(torch.flatten(fake_indices).long(), BATCH_SIZE_TEST)
    decoded = vae_model.decoder(quantised).cpu().detach()
    save_image(decoded, fp='Fake Image.png', normalize=True)

    best_ssim = 0
    best_ssim_image = None
    for image, _ in train_loader:
        ssim = SSIM(image, decoded)
        if ssim > best_ssim:
            best_ssim_image = image
            best_ssim = ssim

    save_image(best_ssim_image, fp='Real Image.png', normalize=True)

if __name__ == '__main__':
    main()
