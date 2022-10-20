__author__ = "Utkarsh Sharma"

import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from modules import VQVAE
from constants import DEVICE
from torchmetrics import StructuralSimilarityIndexMeasure

class VisualiseVQVAE:
    """
    Class which has all of the predict and visualisation functions for VQ-VAE in it.
    Params:
        vqvae_model : trained VQ-VAE model.
    """

    def __init__(self, vqvae_model):
        self.vqvae = vqvae_model

    def real_codebook_reconstructed(self, test_dataloader):
        """
        Shows 5 examples from the test loader of the input image, codebook vector quantisation and the reconstructed
        image.
        """
        fig, axs = plt.subplots(5, 3, figsize=(10, 15))
        for i in range(0, 5):
            test_real = next(iter(test_dataloader))  # load some from test data loader
            test_real = test_real[i]
            test_real = test_real.to(DEVICE).expand((1, 1, 256, 256))
            pre_conv = (self.vqvae.pre_vq_conv(self.vqvae.encoder(test_real))).expand((1, 64, 64, 64))
            _, test_quantized, _, z = self.vqvae.vq_vae(pre_conv)
            test_reconstructions = self.vqvae.decoder(test_quantized)  # reconstructed image
            indices = z.view(64, 64)  # z from VQ-VAE output is the indices of the quantisation
            indices = indices.to('cpu')
            indices = indices.detach().numpy()

            test_reconstructions = test_reconstructions.detach().cpu().numpy()[0]
            axs[i, 0].imshow(np.squeeze(np.transpose(np.squeeze(test_real.to('cpu'), axis=0), (1, 2, 0))),
                             interpolation='nearest')
            axs[i, 0].set_title('Original')
            axs[i, 1].imshow(indices, interpolation='nearest')
            axs[i, 1].set_title('Codebook')
            axs[i, 2].imshow(np.squeeze(np.transpose(test_reconstructions, (1, 2, 0))), interpolation='nearest')
            axs[i, 2].set_title('Reconstructed')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.show()

    def mean_ssim_vqvae(self, test_loader):
        """
        Code to calculate the mean SSIM over the test set.
        """
        total_ssim = 0
        with torch.no_grad():
            for i in range(len(test_loader)):  # known length of test dataset
                data = next(iter(test_loader)).to(DEVICE)
                self.vqvae.eval()

                # Generate prediction
                _, prediction, _ = self.vqvae(data)
                total_ssim += ssim(prediction, data, data_range=1.0)
        return (total_ssim / len(test_loader)).item()


class PredictGAN:
    """
    Class which has all of the predict and visualisation functions for GAN in it.
    Params:
        dc_generator : trained DCGAN generator.
        vqvae_model : trained VQ-VAE model.
    """

    def __init__(self, dc_generator, vqvae_model: VQVAE):
        self.generator = dc_generator
        self.vqvae = vqvae_model

    def generate_discrete_indices(self, unique_values):
        """
        Function to generate the discrete codebook indices from the trained generator.
        Params:
            unique_values : the indices of the unique values from the VQ-VAE.
        """
        # Generates random discrete indices from GAN
        random_noise = torch.randn(1, 100, 1, 1).to(DEVICE)  # generates random noise
        with torch.no_grad():
            sample = self.generator(
                random_noise)  # generates the continuous codebook using the generator and random noise

        # Visualise generated codebook index
        generated_indexes = sample[0][0]
        generated_indexes = torch.flatten(generated_indexes)

        # takes the continuous output from the generator and snaps them to the unique discrete values from the
        # VQ-VAE
        cont_index_min = torch.min(generated_indexes)
        cont_index_max = torch.max(generated_indexes)

        num_intervals = len(unique_values)
        interval_size = (cont_index_max - cont_index_min) / num_intervals

        for i in range(0, num_intervals):
            MIN = cont_index_min + i * interval_size
            generated_indexes[
                torch.logical_and(MIN <= generated_indexes, generated_indexes <= (MIN + interval_size))] = \
                unique_values[i]  # maps all the values in the interval to the discrete mapping for decoder

        snapped_indices = generated_indexes.view(64, 64)
        snapped_indices = snapped_indices.to('cpu')
        snapped_indices = snapped_indices.detach().numpy()
        plt.imshow(snapped_indices)
        plt.show()
        return generated_indexes

    def generate_decoder_sample(self, discrete_indices):
        """
        Function to take the discrete codebook indices and pass them through the decoder to get the final generated
        image.
        """
        generated_index = discrete_indices.long()
        generated_output = self.vqvae.vq_vae.get_quantized(generated_index)
        generated_output = self.vqvae.decoder(generated_output)

        # Visualise the generated image
        decoded_output = generated_output[0][0]
        decoded_output = decoded_output.to('cpu')
        plt.imshow(decoded_output.detach().numpy())
        plt.show()
        return decoded_output

    def mean_ssim_test_set(self, test_loader, generated_sample):
        """
        Function to calculate the mean SSIM between all the images in the test loader and the given generated_sample.
        """
        thresholded_ssim_count = 0
        total_ssim = 0
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        total_samples = 0
        generated_sample = generated_sample.to(DEVICE)[None, None, :, :]
        with torch.no_grad():
            for i in range(len(test_loader)):  # length of test loader
                data = next(iter(test_loader)).to(DEVICE)
                total_samples += data.shape[0]
                self.vqvae.eval()
                # iterate over each example in the batch
                for j in range(data.shape[0]):
                    data_new = (data[j][0])[None, None, :, :]
                # SSIM between the generated sample and the test dataset image
                    ssim_generated = ssim(generated_sample, data_new)
                    if ssim_generated > 0.6:
                        thresholded_ssim_count += 1
                    total_ssim += ssim_generated

        return (total_ssim / total_samples).item(), thresholded_ssim_count
