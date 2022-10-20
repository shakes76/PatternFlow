import torch


def get_noise_cadence():
    """
    Generates a tensor describing the cadence of noise addition timestamps

    Returns:
        Tensor: noise to be added
    """
    return torch.linspace(1e-4, 0.02, 1000)

def add_noise(x, pos):
    """
    Adds noise to tensor x through a noising timeline

    Args:
        x (Tensor): Tensor representation of an image to noise
        pos (int): Position in the noising process to sample beta

    Returns:
        [type]: [description]
    """
    beta = get_noise_cadence().to("cuda")
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[pos])[:, None, None, None]
    sqrt_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[pos])[:, None, None, None]
    E = torch.randn_like(x)
    return sqrt_alpha_cumprod * x + sqrt_minus_alpha_cumprod * E, E

def get_sample_pos(size):
    """
    Generates sampling tensor from input size and predefined sample range

    Args:
        size (int): size to time step

    Returns:
        Tensor: tensor full of random integers between 1 and 1000 with specified size
    """
    return torch.randint(low=1, high=1000, size=(size,))
