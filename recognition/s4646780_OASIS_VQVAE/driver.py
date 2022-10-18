"""
Main functionality which calls all the necessary scripts for model to run.
"""

from train import train_VQVAE
from predict import VisualiseVQVAE


def main():
    # training and visualizing VQVAE model outputs
    vq_vae, errors, test_loader = train_VQVAE()
    visualiseVQVAE = VisualiseVQVAE(vq_vae)
    visualiseVQVAE.makegrid_reconstructed(test_loader)
    print(f"average SSIM of the reconstructed images versus real image is "
          f"{visualiseVQVAE.mean_ssim_vqvae(test_loader)}")



main()