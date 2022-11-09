"""
This file is to test the performance of the generator.
:param path: a path of a checkpoint, this is obligatory
:param --size: the resolution of the images
    e.g.: python test.py checkpoint/train_step-7 --size 256
:return: two grid images: with/without style mixing.
NOTE:
train_step-2 will generate images with resolution = 8
train_step-3 will generate images with resolution = 16
train_step-4 will generate images with resolution = 32
The step should match the resolution, otherwise the images will be weird.
"""

import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from model import StyledGenerator
import math
import argparse


@torch.no_grad()
def get_mean_style(generator, device):
    """
    Sample 1024 latent codes, put them into mapping network and get a mean,
    repeat 10 times, and return a mean style.
    :param generator: the generator of StyleGAN
    :param device: cuda or cpu
    :return: mean style
    """
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    """
    Sample images without style mixing.
    :param generator: the generator of StyleGAN
    :param step: resolution stage. e.g step=7, resolution=256
    :param mean_style: a mean style only through mapping network
    :param n_sample: number of sample images
    :param device: cuda or cpu
    :return: n_sample images generate by generator
    """
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,   # by decreasing style_weight, truncation can be increased
    )

    return image


@torch.no_grad()
def style_mixing(generator, step, mean_style, n_latent0, n_latent1, device):
    """
    Mix Regularization. Mix style from two latent codes
    :param generator: the generator of StyleGAN
    :param step: resolution stage. e.g. resolution=8 >>> step=1, resolution=16 >>> step=2
    :param mean_style: a mean style only through mapping network
    :param n_latent0: number of latent code 0
    :param n_latent1: number of latent code 1
    :param device: cuda or cpu
    :return: a list of images mixing the style from latent code 0 and latent code 1
    """
    latent_code0 = torch.randn(n_latent0, 512).to(device)
    latent_code1 = torch.randn(n_latent1, 512).to(device)

    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]   # a black image
    # by decreasing style_weight, truncation can be increased
    latent0_image = generator(
        latent_code0, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    latent1_image = generator(
        latent_code1, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(latent0_image)

    for i in range(n_latent1):
        image = generator(
            [latent_code1[i].unsqueeze(0).repeat(n_latent0, 1), latent_code0],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,   # by decreasing style_weight, truncation can be increased
            mixing_range=(0, 1),    # low resolutions will mix features from latent_code0
        )
        images.append(latent1_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images


def show(imgs, title):
    """
    Display images
    :param imgs: a list of tensor images
    :param title: str, the title of the images
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    plt.title(title)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test StyleGAN')
    parser.add_argument('path', type=str, help='path to checkpoint')
    parser.add_argument('--size', type=int, default=128, help='the resolution of images')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    generator = StyledGenerator(512).to(device)
    ckpt = torch.load(args.path, map_location=device)
    generator.load_state_dict(ckpt['g_running'])    # g_running is the shadow generator which is more stable.
    generator.eval()

    mean_style = get_mean_style(generator, device)
    step = int(math.log(args.size, 2)) - 2
    n_row, n_col = 1, 5
    img_sample = sample(generator, step, mean_style, n_row*n_col, device)
    save_image(img_sample, 'Sample.png', nrow=n_col, normalize=True, range=(-1, 1))
    sample_grid = make_grid(img_sample, nrow=n_col, normalize=True, value_range=(-1, 1))
    show(sample_grid, 'Sample')

    img_mix = style_mixing(generator, step, mean_style, n_col, n_row, device)
    save_image(img_mix, 'sample_mixing.png', nrow=n_col + 1, normalize=True, range=(-1, 1))
    grid_mix = make_grid(img_mix, nrow=n_col+1, normalize=True, value_range=(-1, 1))
    show(grid_mix, 'mix regularization')


