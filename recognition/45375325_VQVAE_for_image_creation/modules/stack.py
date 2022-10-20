import torch.nn as nn


class ResidualStack(nn.Module):
    """
    A stack of residual layers
    """

    def __init__(self, input_dim, between_latent_dim, latent_dim):
        super(ResidualStack, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=input_dim,
                      out_channels=between_latent_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=between_latent_dim,
                      out_channels=latent_dim,
                      kernel_size=1,
                      stride=1,
                      bias=False
                      )
        )

    def forward(self, x):
        return x + self.net(x)
