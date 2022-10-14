import torch.nn as nn


class ResidualLayer(nn.Module):
    """
    One residual layer
    """

    def __init__(self, input_dim, latent_dim, residuals_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim, residuals_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(residuals_dim, latent_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x += self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers
    """

    def __init__(self, input_dim, latent_dim, residuals_dim, num_stacked_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = num_stacked_layers
        self.stack = nn.ModuleList([ResidualLayer(input_dim, latent_dim, residuals_dim)] * num_stacked_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = nn.functional.relu(x)
        return x
