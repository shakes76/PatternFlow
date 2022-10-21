from time import time
import torch
import torch.nn.functional as F
import csv


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
        Tensor: input tensor with noise added to it
    """
    beta = get_noise_cadence().to("cuda")
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[pos])[:, None, None, None]
    sqrt_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[pos])[:, None, None, None]
    E = torch.randn_like(x)
    return sqrt_alpha_cumprod * x + sqrt_minus_alpha_cumprod * E, E

def remove_noise(img, timestep, model):
    beta = get_noise_cadence().to("cuda")
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    alpha_cumprod_rev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alpha_reciprocal = torch.sqrt(1.0 / alpha)
    sqrt_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    sqrt_minus_alpha_cumprod_x = extract_index(sqrt_minus_alpha_cumprod, timestep, img.shape)
    sqrt_alpha_reciprocal_x = extract_index(sqrt_alpha_reciprocal, timestep, img.shape)

    mean = sqrt_alpha_reciprocal_x * (img - extract_index(beta, timestep, img.shape) * model(img, timestep) / sqrt_minus_alpha_cumprod_x)


    if timestep == 0:
        return mean
    else:
        E = torch.randn_like(img)
        posterior_variance = beta * (1. - alpha_cumprod_rev) / (1.0 - alpha_cumprod)

        return mean + torch.sqrt(extract_index(posterior_variance, timestep, img.shape)) * E

def get_sample_pos(size):
    """
    Generates sampling tensor from input size and predefined sample range

    Args:
        size (int): size to time step

    Returns:
        Tensor: tensor full of random integers between 1 and 1000 with specified size
    """
    return torch.randint(low=1, high=1000, size=(size,))

def extract_index(x, pos, x_shape):
    """
    Returns a specific index, pos, in a tensor, x

    Args:
        x (Tensor): input tensor
        pos (Tensor): position tensor
        x_shape (Size): shape of tensor

    Returns:
        Tensor: index tensor
    """
    batch_size = pos.shape[0]
    output = x.gather(-1, pos.to("cuda"))
    output = output.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to("cuda")
    return output

def save_loss_data(tracked_loss, test_loss):
    """
    Save training and testing loss data to csv file

    Args:
        tracked_loss (List): list of training loss values
        test_loss (List): list of testing loss values
    """
    # Save loss values
    train_loss_file = open("Epoch Loss.csv", 'w')
    writer = csv.writer(train_loss_file)
    writer.writerows(tracked_loss)

    # Save test values
    test_loss_file = open("Test Loss.csv", 'w')
    writer = csv.writer(test_loss_file)
    writer.writerows(test_loss)
