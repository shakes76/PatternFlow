import torch.nn as nn
from stack import ResidualStack


class Decoder(nn.Module):
    """
    Decoder network

    given a latent sample z, maps to the original space
    """

    def __init__(self, input_dim, latent_dim, num_residual_layers, num_stacked_layers):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_steps = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim, latent_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualStack(latent_dim, latent_dim, num_stacked_layers, num_residual_layers),
            nn.ConvTranspose2d(
                latent_dim, latent_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_steps(x)
