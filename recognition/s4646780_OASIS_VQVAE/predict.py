import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pytorch_msssim import ssim


class VisualiseVQVAE:
    """

    """

    def __init__(self, vqvae_model):
        self.vqvae = vqvae_model

    def real_codebook_reconstructed(self, test_dataloader):
        """

        :param test_dataloader:
        :return:
        """
        fig, axs = plt.subplots(5, 3, figsize=(10, 15))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(0, 5):
            test_real = next(iter(test_dataloader))  # load some from test dl
            test_real = test_real[i]
            test_real = test_real.to(device)
            pre_conv = (self.vqvae.pre_vq_conv(self.vqvae.encoder(test_real))).expand((1, 64, 64, 64))
            _, test_quantized, _, z = self.vqvae.vq_vae(pre_conv)
            test_reconstructions = self.vqvae.decoder(test_quantized)
            indices = z.view(64, 64)
            indices = indices.to('cpu')
            indices = indices.detach().numpy()

            npimg = test_reconstructions.detach().numpy()[0]
            axs[i, 0].imshow(np.squeeze(np.transpose(test_real, (1, 2, 0))), interpolation='nearest')
            axs[i, 0].set_title('Original')
            axs[i, 1].imshow(indices, interpolation='nearest')
            axs[i, 1].set_title('Codebook')
            axs[i, 2].imshow(np.squeeze(np.transpose(npimg, (1, 2, 0))), interpolation='nearest')
            axs[i, 2].set_title('Reconstructed')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.show()

    def mean_ssim_vqvae(self, test_loader):
        """
        Code to calculate the mean SSIM over the test set.
        """
        total_ssim = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for i in range(544):  # known length of test dataset
                data = next(iter(test_loader)).to(device)
                self.vqvae.eval()

                # Generate prediction
                _, prediction, _ = self.vqvae(data)
                total_ssim += ssim(prediction, data, data_range=255)
        return (total_ssim / 554).item()


class PredictGAN:
    def __init__(self, dc_generator, vqvae_model):
        self.generator = dc_generator
        self.vqvae = vqvae_model

    def generate_discrete_indices(self, unique_values):
        # Generates random discrete indices from GAN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random_noise = torch.randn(1, 100, 1, 1).to(device)  # generates random noise
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
        return generated_indexes

    def generate_decoder_sample(self, discrete_indices):
        generated_index = discrete_indices.long()
        generated_output = self.vqvae.vq_vae.get_quantized(generated_index)
        generated_output = self.vqvae.decoder(generated_output)

        # Visualise
        decoded_output = generated_output[0][0]
        decoded_output = decoded_output.to('cpu')
        decoded_output = decoded_output.detach().numpy()
        plt.imshow(decoded_output)
        return decoded_output

    def mean_ssim_test_set(self,  test_loader, generated_sample):
        """
        Code to calculate the mean SSIM over the test set.
        """
        thresholded_ssim_count = 0
        total_ssim = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for i in range(554):
                data = next(iter(test_loader)).to(device)
                self.vqvae.eval()

                # SSIM between the generated sample and the test dataset
                ssim_generated = ssim(generated_sample, data, data_range=1.0)
                if ssim_generated > 0.6:
                    thresholded_ssim_count += 1
                total_ssim += ssim_generated

        return (total_ssim / 554).item(), thresholded_ssim_count
