import argparse
import random
import math

from data_loader import MultiResolutionDataset
from network import Styled_G, D

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from PIL import Image

def stacking_parameters(net_0, net_1, weight_decay=0.999):
    """Accumulate the parameters of two models based on the weight decay"""
    parameter_0, parameter_1 = dict(net_0.named_parameters()),\
                               dict(net_1.named_parameters())

    for key in parameter_0.keys():
        parameter_0[key].data.mul_(weight_decay).add_(1 - weight_decay,
                                                    parameter_1[key].data)

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult



def train(args, generator, discriminator):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(G_optimiser, args.lr.get(resolution, 0.001))
    adjust_lr(D_optimiser, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(60_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': G_optimiser.state_dict(),
                    'd_optimizer': D_optimiser.state_dict(),
                    'g_running': G_processing.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(G_optimiser, args.lr.get(resolution, 0.001))
            adjust_lr(D_optimiser, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, latent_length, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, latent_length, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        D_optimiser.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            if i%10 == 0:
                gen_loss_val = loss.item()

            loss.backward()
            G_optimiser.step()
            stacking_parameters(G_processing, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        G_processing(
                            torch.randn(gen_j, latent_length).cuda(), step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                G_processing.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader

if __name__ == '__main__':

    """ Set up the experimental arguments """
    batch_size, latent_length, n_critic = 16, 512, 1
    parser = argparse.ArgumentParser(description='StyleGAN')
    parser.add_argument('path', type=str, help='Dataset Path ')
    parser.add_argument('--phase', type=int, default=40_000,
                        help='number of samples for each training phase')
    parser.add_argument('--init_size', default=8, type=int,
                        help='initial size of input images')
    parser.add_argument('--sched', action='store_true',
                        help='scheduling for lr')
    parser.add_argument('--max_size', default=512, type=int,
                        help='max size of generated images')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoints')
    parser.add_argument( '--no_from_rgb_activate', action='store_true',
                         help='activate in from_rgb')
    parser.add_argument( '--mixing', action='store_true',
                         help='mixing regularization')
    parser.add_argument( '--loss', type=str, default='wgan-gp',
                         choices=['wgan-gp', 'r1'], help='choose gan loss')
    args = parser.parse_args()

    """ Load the Pytorch networks of both generator and discriminator """
    G_net = nn.DataParallel(Styled_G(latent_length)).cuda()
    D_net = nn.DataParallel(
        D(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    G_processing = Styled_G(latent_length).cuda()
    G_processing.train(False)

    """ Set up optimisor for Pytorch training """
    beta_0 = 0.0
    beta_1 = 0.99
    G_optimiser = optim.Adam(
        G_net.module.generator.parameters(), lr=args.lr, betas=(beta_0, beta_1)
    )
    G_optimiser.add_param_group(
        {
            'params': G_net.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    D_optimiser = optim.Adam(D_net.parameters(), lr=args.lr,
                             betas=(beta_0, beta_1))

    """ Start Training """
    train(args, G_net, D_net)

    """ End of training """
    print("StyleGAN Training Complete")