import random
import math
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.nn import functional as F
from data_loader import MultiResolutionDataset
from network import Styled_G, D
# import all required libraries.


def learning_rate_decay(optim, learning_rate):
    """
    The learning rate decay is used to multiplicate the learning rate with a 
    multiplier in the optimser. 
    """
    index = 1
    for prameters in optim.param_groups:
        prameters['lr'] = learning_rate * prameters.get('mult', index)


def stacking_parameters(net_0, net_1, weight_decay=0.999):
    """
    Accumulate the parameters of two models according to the weight decay.
    Soecifically, the weight of net_1 is stacked into the net_2, with
    the stacked weight are multipled with 1-weight_decay
    """
    parameter_0, parameter_1 = dict(net_0.named_parameters()),\
                               dict(net_1.named_parameters())

    for key in parameter_0.keys():
        parameter_0[key].data.mul_(weight_decay).add_(1 - weight_decay,
                                                    parameter_1[key].data)


def train_StyleGAN(args, G, D):
    """
    Image transformation and data augmentation
    including random flip, and normalisation.
    """
    norm_index = 0.5
    image_transformation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((norm_index, norm_index, norm_index), 
                (norm_index, norm_index, norm_index), inplace=True),
        ]
    )

    """
    Set up the dataset
    """
    dataset = MultiResolutionDataset(args.path, image_transformation)
    # set the batch size and the learning rate manually 
    args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
    args.gen_sample = {512: (8, 4), 1024: (4, 2)}
    args.batch_default = 32

    """
    progressive increase the dimension size.
    """
    progressive_stage = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** progressive_stage

    """
    Load data set with different resolution as preprocesssed. 
    Each resolution contains all the input images at that dimension.  
    """
    loaded_dataset = shuffle_samples(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loaded_dataset) # iterate the loaded dataset for further training

    """
    learning rate decay is used for optimisers of both generator and discriminator.
    Because the reduced weights during training can make the optimiser converge 
    more stable near the end of the training. 
    """
    learning_rate_decay(G_optimiser, args.lr.get(resolution, 0.001))
    learning_rate_decay(D_optimiser, args.lr.get(resolution, 0.001))

    # for visulise the training process
    pbar = tqdm(range(60_000))

    """
    At the begining of the training, only the discriminator is trained, 
    thus, the change_gradient_status() function is called to stop train the
    generator.
    """
    change_gradient_status(G, False)
    change_gradient_status(D, True)

    """
    Initialise the error/loss of the discriminator, generator and the gradient.
    These three varialbles will be used for further calculation and printed 
    to the terminel. 
    """
    D_error, G_error, gradient_error = 0, 0, 0

    """
    rocessed data is used to record the data that has been used to train the model
    """
    processed_data, alpha = 0, 0

    """
    compute the max stage number according to the max size
    """
    most_progressive_stage = int(math.log2(args.max_size)) - 2
    """
    at the begining and mid stage, the last_progressive_stage should be false
    """
    last_progressive_stage = False

    """
    start to train the styleGAN
    """
    for index in pbar:

        D.zero_grad() # clear the gradient from last iteration

        # compute alpha to control model learning
        alpha = min(1, 1 / args.phase * (processed_data + 1))

        if (resolution == args.init_size and args.ckpt is None) or last_progressive_stage:
            alpha = 1

        """
        if the processed data is larger than 2 times of phase number,
        then the progressive_stage is increased, which means the resolution 
        of the feature map increase.
        """
        if processed_data > args.phase * 2:
            processed_data = 0
            progressive_stage += 1

            # limit the progressive stage to be larger than
            # the max stage
            if progressive_stage > most_progressive_stage:
                progressive_stage = most_progressive_stage
                last_progressive_stage = True
                ckpt_step = progressive_stage + 1
            else:
                alpha = 0
                ckpt_step = progressive_stage

            # determin the resolution at current stage
            resolution = 4 * 2 ** progressive_stage

            loaded_dataset = shuffle_samples(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loaded_dataset)

            # for each progressive stage, the checkpoint is saved,
            # for further generating images.
            torch.save(
                {
                    'generator': G.module.state_dict(),
                    'discriminator': D.module.state_dict(),
                    'g_optimizer': G_optimiser.state_dict(),
                    'd_optimizer': D_optimiser.state_dict(),
                    'g_running': G_processing.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            # Each iteration, the learning rate is reduced.
            learning_rate_decay(G_optimiser, args.lr.get(resolution, 0.001))
            learning_rate_decay(D_optimiser, args.lr.get(resolution, 0.001))

        try:
            real_AKOA_data = next(data_loader) # load real images
        except (OSError, StopIteration): # prevent system error
            data_loader = iter(loaded_dataset)
            real_AKOA_data = next(data_loader)

        processed_data += real_AKOA_data.shape[0] # count the processed data
        batch_size = real_AKOA_data.size(0) # obtain the batch size
        real_AKOA_data = real_AKOA_data.cuda() # load the data to cuda

        """
        Two different losses are provided here for different datasets,
        based on the original implementation.
        But for AKOA, we only tried wgan-gp as the time is not enough.
        """
        if args.loss == 'wgan-gp':
            output_real = D(real_AKOA_data, step=progressive_stage, alpha=alpha)
            output_real = output_real.mean() - 0.001 * (output_real ** 2).mean()
            (-output_real).backward()
        else:
            print('r1 loss is not implemented')
            exit()

        """
        Implementation of the style mixing module here.
        """
        if args.mixing and random.random() < 0.9:
            w_11, w_12, w_21, w_22 = torch.randn(
                4, batch_size, latent_length, device='cuda'
            ).chunk(4, 0)
            w_0 = [w_11.squeeze(0), w_12.squeeze(0)]
            w_1 = [w_21.squeeze(0), w_22.squeeze(0)]
        else: # no mixing is requied, always this branch
            w_0, w_1 = torch.randn(2, batch_size, latent_length, device='cuda').chunk(
                2, 0
            )
            w_0, w_1 = w_0.squeeze(0), w_1.squeeze(0)

        """
        Generate fake images using the generator
        and the generated images are fed into the discriminator
        """
        data_fake = G(w_0, step=progressive_stage, alpha=alpha)
        output_fake = D(data_fake, step=progressive_stage, alpha=alpha)

        """
        Apply the loss function for the gradient backward using the wgan-gp
        """
        if args.loss == 'wgan-gp':
            output_fake = output_fake.mean()
            output_fake.backward()
            eps = torch.rand(batch_size, 1, 1, 1).cuda()
            x_hat = eps * real_AKOA_data.data + (1 - eps) * data_fake.data
            x_hat.change_gradient_status = True
            hat_predict = D(x_hat, step=progressive_stage, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if index%10 == 0:
                gradient_error = grad_penalty.item()
                D_error = (-output_real + output_fake).item()

        else:
            print('r1 is not supported in this code')

        D_optimiser.step() # pass the gradient


        """
        Now, start to train the generator.
        The process below os similar to above, except the
        gradient of the generator is not fixed,
        but the discriminator is fixed.
        """
        if (index + 1) % n_critic == 0:
            G.zero_grad()
            change_gradient_status(G, True)
            change_gradient_status(D, False)

            data_fake = G(w_1, step=progressive_stage, alpha=alpha)
            output = D(data_fake, step=progressive_stage, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -output.mean()
            elif args.loss == 'r1':
                loss = F.softplus(-output).mean()
            if index%10 == 0:
                G_error = loss.item()
            loss.backward()
            G_optimiser.step()

            """
            using the parameters not only in this iteration,
            but also the historical parameters.
            """
            stacking_parameters(G_processing, G.module)
            change_gradient_status(G, False)
            change_gradient_status(D, True)
            """
            Complete the training for 1 iteration.
            """
        
        """
        Save images every 100 iterations. 
        """
        if (index + 1) % 100 == 0:
            generated_AKOA_images = []
            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))
            with torch.no_grad():
                for _ in range(gen_i):
                    generated_AKOA_images.append(
                        G_processing(
                            torch.randn(gen_j, latent_length).cuda(), step=progressive_stage, alpha=alpha
                        ).data.cpu()
                    )
            utils.save_image(
                torch.cat(generated_AKOA_images, 0),
                f'sample/{str(index + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        """
        save the checkpoint every 10000 iterations. 
        """
        if (index + 1) % 10000 == 0:
            torch.save(
                G_processing.state_dict(), f'checkpoint/{str(index + 1).zfill(6)}.model'
            )
            
        """
        print some messages when training the network to inform.
        """
        state_msg = (
            f'Size: {4 * 2 ** progressive_stage}; G: {G_error:.3f}; D: {D_error:.3f};'
            f' Grad: {gradient_error:.3f}; Alpha: {alpha:.5f}'
        )
        pbar.set_description(state_msg)



def change_gradient_status(network, status=True):
    """
    This function is used to change the status of the gradient 
    for a specific network. It changes the gradient status
    of all the parameters in a network. If the status is false, then
    the network is fixed and no gradient is passed backward.
    """
    for prameters in network.parameters():
        prameters.requires_grad = status

def shuffle_samples(samples, batch_size, feature_map_size=4):
    """
    This function is used to shuffle the whole dataset, given the
    specific batch size and feature_map_size. The final sampled dataset
    is also loaded into the DataLoader, which will be used for later
    Pytorch usage.
    """
    samples.resolution = feature_map_size
    return DataLoader(samples, shuffle=True, 
        batch_size=batch_size, num_workers=1, drop_last=True)

if __name__ == '__main__':

    """ Set up the experimental arguments """
    batch_size, latent_length, n_critic = 16, 512, 1
    parser = argparse.ArgumentParser(description='StyleGAN')
    parser.add_argument('path', type=str, help='Dataset Path ')
    parser.add_argument('--phase', type=int, default=40_000,
                        help='number of samples for each training phase')
    parser.add_argument('--init_size', default=8, type=int,
                        help='initial size of input images')
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
    train_StyleGAN(args, G_net, D_net)

    """ End of training """
    print("StyleGAN Training Complete")
