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

    def makegrid_reconstructed(self, test_dataloader):
        """

        :param test_dataloader:
        :return:
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_real = next(iter(test_dataloader))  # continually load images from test set data loader
        test_real = test_real[0]
        test_real = test_real.to(device)
        pre_conv = (self.vqvae.pre_vq_conv(self.vqvae.encoder(test_real))).expand((1, 64, 64, 64))  # encoder, reshape
        _, test_quantized, _, _ = self.vqvae.vq_vae(pre_conv)
        test_reconstructions = self.vqvae.decoder(test_quantized)

        def show(img, image_type):
            np_img = img.detach().numpy()
            fig = plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            fig.set_title(image_type)

        # show reconstructed images
        show(make_grid(test_reconstructions.cpu(), nrow=4), "Reconstructed Image")
        show(make_grid(test_real.cpu(), nrow=4), "Real Image")

    def real_codebook_reconstructed(self, test_dataloader):
        """

        :param test_dataloader:
        :return:
        """
        pass

    def mean_ssim_vqvae(self, test_loader):
        """
        Code to calculate the mean SSIM over the test set.
        """
        total_ssim = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for i in range(544): # known length of test dataset
                data = next(iter(test_loader)).to(device)
                self.vqvae.eval()

                # Generate prediction
                _, prediction, _ = self.vqvae(data)
                total_ssim += ssim(prediction, data, data_range=255)
        return (total_ssim / 554).item()


class PredictGan:
    def __init__(self, dcgan_model):
        self.gan = dcgan_model
