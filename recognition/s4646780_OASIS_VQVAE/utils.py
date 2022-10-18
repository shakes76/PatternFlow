import torch
from pytorch_msssim import ssim
import torch.nn as nn


def mean_ssim_vqvae(test_dataset, test_loader, vqvae_model):
    """
    Code to calculate the mean SSIM over the test set.
    """
    total_ssim = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            data = next(iter(test_loader)).to(device)
            vqvae_model.eval()

            # Generate prediction
            _, prediction, _ = vqvae_model(data)
            total_ssim += ssim(prediction, data, data_range=255)
    return (total_ssim / len(test_dataset)).item()


def initialise_weights(model):
    """
    Function to initialize weights for GAN as specified in original GAN paper.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
