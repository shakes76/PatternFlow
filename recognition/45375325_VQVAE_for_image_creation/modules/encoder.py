import torch.nn as nn
from stack import ResidualStack


class Encoder(nn.Module):
    """
    Encoding network

    given data x, maps the data to latent space
    """

    def __init__(self, input_dim, latent_dim, num_residual_layers, num_stacked_layers):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_steps = nn.Sequential(
            nn.Conv2d(input_dim, latent_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(latent_dim, latent_dim, num_residual_layers, num_stacked_layers)
        )

    def forward(self, x):
        return self.conv_steps(x)
