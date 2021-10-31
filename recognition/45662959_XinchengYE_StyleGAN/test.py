import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import StyledGenerator
import math
import argparse


@torch.no_grad()
def get_mean_style(generator, device):
    """

    :param generator: the generator of StyleGAN
    :param device: cuda or cpu
    :return:
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
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,   # by decreasing style_weight, truncation can be increased
    )

    return image


@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    """

    :param generator: the generator of StyleGAN
    :param step: resolution stage. e.g. resolution=8 >>> step=1, resolution=16 >>> step=2
    :param mean_style:
    :param n_source:
    :param n_target:
    :param device: cuda or cpu
    :return:
    """
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)

    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]   # a black image
    # by decreasing style_weight, truncation can be increased
    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,   # by decreasing style_weight, truncation can be increased
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images


def show(imgs):
    """
    Display images
    :param imgs: a list of tensor images
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
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
    generator.load_state_dict(ckpt['g_running'])
    generator.eval()

    mean_style = get_mean_style(generator, device)
    step = int(math.log(args.size, 2)) - 2
    img_sample = sample(generator, step, mean_style, 5, device)
    sample_grid = make_grid(img_sample)
    show(sample_grid)

