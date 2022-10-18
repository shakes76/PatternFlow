import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class VisualiseVqvae:
    def __init__(self, vqvae_model):
        self.vqvae = vqvae_model

    def makegrid_reconstructed(self, test_dataloader):
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


class VisualiseGan:
    def __init__(self, dcgan_model):
        self.gan = dcgan_model
