# !/user/bin/env python
"""
The script trains the StyleGAN
"""

from train import Trainer
import argparse

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


# parse commandline inputs
parser = argparse.ArgumentParser(description='Train the generative model to produce high resolution images')
parser.add_argument("data_dir", type=str, help="folder of the training data")
parser.add_argument("output_dir", type=str, help="output folder")
parser.add_argument("g_input_res", type=int,
                    help="resolution of the first convolutional layer in the generator")
parser.add_argument("g_init_filters", type=int,
                    help="number of filters of the first convolutional layer in the generator")
parser.add_argument("d_final_res", type=int,
                    help="output resolution of the last convolutional layer in the discriminator")
parser.add_argument("d_input_filters", type=int,
                    help="number of filters of the first convolutional layer in the discriminator")
parser.add_argument("fade_in_base", type=float,
                    help="the divisor of the current epoch number when calculating the fade in factor")
# optional inputs
parser.add_argument("--resolution", default=64, type=int, help="the resolution of the output images, defaults to 64")
parser.add_argument("--channels", default=1, type=int, help="number of channels, defaults to 1")
parser.add_argument("--latent", default=100, type=int, help="the length of the input latent, defaults to 100")
parser.add_argument("--batch", default=128, type=int, help="batch size, defaults to 128")
parser.add_argument("--epochs", default=20, type=int, help="number of training epochs, defaults to 20")
parser.add_argument("--checkpoint", default=1, type=int, help="save frequency in number of epochs, defaults to 1")
parser.add_argument("--lr", default=0.0002, type=float, help="learning rate of the optimizers, defaults to 0.0002")
parser.add_argument("--beta", default=0.5, type=float,
                    help="exponential decay rate for the first moment estimate, defaults to 0.5")
parser.add_argument("--val", default=16, type=int, help="number of validation images")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("-n", "--neptune",
                    help="whether to use Neptune to track the training metrics", action="store_true")

args = parser.parse_args()

data = args.data_dir
output_dir = args.output_dir
g_input_res = args.g_input_res
g_init_filters = args.g_init_filters
d_final_res = args.d_final_res
d_input_filters = args.d_input_filters
fade_in_base = args.fade_in_base
resolution = args.resolution
channels = args.channels
latent = args.latent
batch = args.batch
epochs = args.epochs
checkpoint = args.checkpoint
lr = args.lr
beta = args.beta
val = args.val
seed = args.seed
neptune = args.neptune

# launch the training
trainer = Trainer(data, output_dir, g_input_res, g_init_filters, d_final_res, d_input_filters, fade_in_base,
                  resolution=resolution, channels=channels, latent_dim=latent, batch=batch, epochs=epochs,
                  checkpoint=checkpoint, lr=lr, beta_1=beta, validation_images=val, seed=seed, use_neptune=neptune)

trainer.train()
