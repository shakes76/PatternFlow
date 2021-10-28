import argparse
import math

import torch
from torchvision import utils

from network import Styled_G


@torch.no_grad()
def average_styles(G, gpu):
    """
    Average all the styles that produced by the generator,
    and return the averaged result. 
    """
    mean_style = None

    for i in range(10):
        style = G.mean_style(torch.randn(1024, 512).to(gpu))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def produce_single_AKOA_image(G, step, style_averaged, number_data, gpu):
    """
    This is method is simply used to call the generator to generate 
    fake AKOA image. 
    """
    return G(
        torch.randn(number_data, 512).to(gpu),
        step=step,
        alpha=1,
        mean_style=style_averaged,
        style_weight=0.7,
    )

@torch.no_grad()
def mixing_regularisation(G, step, style_averaged, s_numers, 
                        num_T, gpu):
    """
    Apply the mixing regularisation as proposed in paper,
    which is to create teo different latent code w_0 and w_1 and 
    input them to the generator seperatly. 
    """
    latent_0 = torch.randn(s_numers, 512).to('cuda')
    latent_1 = torch.randn(num_T, 512).to('cuda')
    
    resolution = 4 * 2 ** step
    a = 1

    AKOA_img = [torch.ones(1, 3, resolution, resolution).to('cuda') * -1]

    fake_0 = G(
        latent_0, step=step, alpha=a, mean_style=style_averaged, style_weight=0.7
    )
    fake_1 = G(
        latent_1, step=step, alpha=a, mean_style=style_averaged, style_weight=0.7
    )

    AKOA_img.append(fake_0)

    for index in range(num_T):
        AKOA_fake_img = G(
            [latent_1[index].unsqueeze(0).repeat(s_numers, 1), latent_0],
            step=step,
            alpha=a,
            mean_style=style_averaged,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        AKOA_img.append(fake_1[index].unsqueeze(0))
        AKOA_img.append(AKOA_fake_img)

    AKOA_img = torch.cat(AKOA_img, 0)
    
    return AKOA_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fake AKOA images using trained generator.')
    parser.add_argument('checkpoint', type=str, help='checkpoint to checkpoint file')
    parser.add_argument('--column_size', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('--row_size', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--dimension', type=int, default=1024, help='the dimension of the generated AKOA image')
    
    args = parser.parse_args()

    """
    Set up the generator, and load the trained weights to it,
    change the status to the evalutation. 
    """
    G = Styled_G(512).to('cuda')
    G.load_state_dict(torch.load(args.checkpoint)['g_running'])
    G.eval()

    mean_style = average_styles(G, 'cuda')

    step = int(math.log(args.dimension, 2)) - 2
    
    img = produce_single_AKOA_image(G, step, mean_style, args.row_size * args.column_size, 'cuda')
    utils.save_image(img, 'sample.png', nrow=args.column_size, normalize=True, range=(-1, 1))
    
    for j in range(20):
        img = mixing_regularisation(G, step, mean_style, args.column_size, args.row_size, 'cuda')
        utils.save_image(
            img, f'sample_mixing_{j}.png', nrow=args.column_size + 1, normalize=True, range=(-1, 1)
        )
