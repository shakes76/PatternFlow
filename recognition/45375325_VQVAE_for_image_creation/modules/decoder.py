import torch.nn as nn
from modules.stack import ResidualStack


class Decoder(nn.Module):
    """
    Decoder network

    given a latent sample z, maps to the original space
    """

    def __init__(self, input_dim, between_latent_dim, latent_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_steps = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=latent_dim,
                      kernel_size=kernel - 1,
                      stride=stride - 1,
                      padding=1
                      ),
            ResidualStack(input_dim=latent_dim,
                          between_latent_dim=between_latent_dim,
                          latent_dim=latent_dim
                          ),
            ResidualStack(input_dim=latent_dim,
                          between_latent_dim=between_latent_dim,
                          latent_dim=latent_dim
                          ),
            nn.ConvTranspose2d(in_channels=latent_dim,
                               out_channels=latent_dim//2,
                               kernel_size=kernel,
                               stride=stride,
                               padding=1
                               ),
            nn.ConvTranspose2d(in_channels=latent_dim//2,
                               out_channels=3,
                               kernel_size=kernel,
                               stride=stride,
                               padding=1
                               )
        )

    def forward(self, x):
        return self.inverse_conv_steps(x)
