from io import BytesIO
import lmdb
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import random
import math
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable, grad
from torch.autograd import Function
from model import StyledGenerator, Discriminator
import time
import datetime
import matplotlib.pyplot as plt

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


def sample_data(dataset, batch_size, resolution=4):
    """
    Get certain images with input resolution from the dataset.
    :param dataset: the whole dataset
    :param batch_size
    :param resolution
    :return: the dataset only containing images with input resolution
    """
    dataset.resolution = resolution
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, drop_last=True)

    return loader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    """
    Exponential Moving Average (EMA) of generator, can improve the robustness
    model1_{t} = decay * model_{t-1} + (1-decay) * model2_{t}
    :param model1: g_running, a shadow generator which will not participate in training directly
    :param model2: generator.module, a real generator used in training
    :param decay: weight, usually [0.9, 0.999]
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

# 18680 images
def train(args, dataset, generator, discriminator):
    """
    Main train loop.
    :param args:
    :param dataset: the whole dataset
    :param generator
    :param discriminator
    """
    print('Starting training loop...')
    step = int(math.log2(args.init_size)) - 2   # training resolution 8:1,16:2 32:3 64:4 128:5 256:6
    resolution = 4 * 2 ** step
    loader = sample_data(dataset, args.batch.get(resolution, args.batch_default), resolution)
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    progress_bar = tqdm(range(3_000_000))   # total epochs over all resolutions

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    disc_list = []
    gen_loss_val = 0
    gen_list = []
    grad_loss_val = 0

    alpha = 0   # interpolation between previous resolutions and new (larger) resolutions
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False
    t0 = time.time()
    # train_time = []

    for i in progress_bar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if not os.path.exists('losses'):
                os.mkdir('losses')
            plt.plot(disc_list, label="discriminator")
            plt.plot(gen_list, label='generator')
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.title(f'Loss for Resolution{step}')
            plt.legend()
            plt.savefig(f'./losses/loss-step{step}.png')

            disc_list = []
            gen_list = []

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
            t = time.time() - t0
            t = datetime.timedelta(seconds=t)
            print('\nrunning time', t)
            # train_time.append(t)
            t0 = time.time()
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        # calculate wgan-gp loss of discriminator to classify a real image
        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        (-real_predict).backward()

        # (F) mixing regularization to localize the style
        # use more than 1 random latent codes for generation
        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        # generate a fake image
        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        # calculate wgan-gp loss of discriminator to classify a fake image and backpropagation
        fake_predict = fake_predict.mean()
        fake_predict.backward()
        eps = torch.rand(b_size, 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        if i % 10 == 0:
            grad_loss_val = grad_penalty.item()
            disc_loss_val = (-real_predict + fake_predict).item()
            disc_list.append(disc_loss_val)
        d_optimizer.step()

        # calculate the loss of generator and backpropagation
        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)
            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()

            if i % 10 == 0:
                gen_loss_val = loss.item()
                gen_list.append(gen_loss_val)

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if not os.path.exists('sample'):
            os.mkdir('sample')

        if (i + 1) % 1000 == 0:
            images = []
            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(), step=step, alpha=alpha
                        ).data.cpu()
                    )
            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')

        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), f'checkpoint/g_running{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; lr: {args.lr.get((4 * 2 ** step), 0.001):.5f}; batch size: {args.batch.get(resolution, args.batch_default)};'
            f'G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f}; Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        progress_bar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    n_critic = 1    # the number of critic (discriminator) iterations per generator iteration
                    # In WGAN-GP paper, they use n_critic = 5

    parser = argparse.ArgumentParser(description='train StyleGAN')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=300_000,    # phase should be large enough, it also controls alpha, otherwise generator will be unstable
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched',
                        action='store_true',
                        help='use lr and batch size scheduling for different resolutions')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    args = parser.parse_args()
    # args.path = "AKOA_PRE"
    # args.ckpt = 'checkpoint/train_step-7'

    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        print('Loading checkpoint...')
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, generator, discriminator)
    print('Completed training on all resolutions')

