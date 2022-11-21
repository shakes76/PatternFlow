"""
Main functionality which calls all the necessary scripts for model to run.
"""
__author__ = "Utkarsh Sharma"

from train import train_VQVAE, train_DCGAN
from predict import VisualiseVQVAE, PredictGAN
from constants import VQVAE_EPOCHS, VQVAE_LEARNING_RATE, GAN_EPOCHS, LEARNING_RATE_GAN, GAN_BATCH_SIZE


def main():
    """
    Main functionality which runs the whole pipeline.
    """
    # training and visualizing VQ-VAE model outputs
    vq_vae, errors, test_loader = train_VQVAE(VQVAE_EPOCHS, VQVAE_LEARNING_RATE)
    visualiseVQVAE = VisualiseVQVAE(vq_vae)
    visualiseVQVAE.real_codebook_reconstructed(test_loader)
    print(f"average SSIM of the reconstructed images versus real image is "
          f"{visualiseVQVAE.mean_ssim_vqvae(test_loader)}")

    # training the DCGAN model using the trained VQ-VAE
    generator, unique_values = train_DCGAN(vq_vae, GAN_EPOCHS, LEARNING_RATE_GAN, GAN_BATCH_SIZE )
    gan_predict = PredictGAN(generator, vq_vae)
    discrete_generated_sample = gan_predict.generate_discrete_indices(unique_values)
    decoded_output = gan_predict.generate_decoder_sample(discrete_generated_sample)
    print(decoded_output.shape)
    avg_ssim_gen_sample, count = gan_predict.mean_ssim_test_set(test_loader, decoded_output)
    print(f'The average ssim of the generated sample is {avg_ssim_gen_sample}. Furthermore, the sample has an SSIM'
          f'>0.6 in {count} out of the 554 possible samples.')


if __name__ == "__main__":
    main()
