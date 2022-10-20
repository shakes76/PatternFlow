import torch.nn as nn
from modules.stack import ResidualStack


class Encoder(nn.Module):
    """
    Encoding network

    given data x, maps the data to latent space
    """

    def __init__(self, input_dim, between_latent_dim, latent_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_steps = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=latent_dim//2,
                      kernel_size=kernel,
                      stride=stride,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_dim//2,
                      out_channels=latent_dim,
                      kernel_size=kernel,
                      stride=stride,
                      padding=1
                      ),
            ResidualStack(input_dim=latent_dim,
                          between_latent_dim=between_latent_dim,
                          latent_dim=latent_dim
                          ),
            ResidualStack(input_dim=latent_dim,
                          between_latent_dim=between_latent_dim,
                          latent_dim=latent_dim
                          )
        )

    def forward(self, x):
        return self.conv_steps(x)
