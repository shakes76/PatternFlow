import os
import argparse
import shutil

import torch
from torch.backends import cudnn

from utils import make_dataset,make_logger, list_dir_recursively_with_ignore, copy_files_and_create_dirs
from model.GAN import StyleGAN

output_dir = '/stylegan/rahinge256'
device = "cuda"
device_id = "0"
resolution = 256
use_ema = True


# Load fewer layers of pre-trained models if possible
def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleGAN pytorch implementation.")
    parser.add_argument('--config', default='./configs/sample.yaml')

    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")

    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--gen_shadow_file", action="store", type=str, default=None,
                        help="pretrained gen_shadow file")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="saved state of generator optimizer")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer")
    args = parser.parse_args()


    # make output dir


    # if os.path.exists(output_dir):
    #     raise KeyError("Existing path: ", output_dir)
    # os.makedirs(output_dir)
    print("copy")
    # copy codes and config file
    files = list_dir_recursively_with_ignore('.', ignores=['diagrams', 'configs'])

    files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]

    # copy_files_and_create_dirs(files)

    # shutil.copy2(args.config, output_dir)
    print("finish copy")
    # logger
    logger = make_logger("project", output_dir, 'log')

    # device
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        num_gpus = len(device_id.split(','))
        logger.info("Using {} GPUs.".format(num_gpus))
        logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
        cudnn.benchmark = True
    device = torch.device(device)

    # create the dataset for training
    dataset = make_dataset("AKOA_Analysis",256)

    # init the network
    style_gan = StyleGAN(structure="linear",
                         resolution= resolution,
                         num_channels= 3,
                         latent_size= 512,
                         loss = "RAhinge",
                         drift=0.001,
                         d_repeats=1,
                         use_ema=True,
                         ema_decay=0.999,
                         device=device)

    # Resume training from checkpoints
    if args.generator_file is not None:
        logger.info("Loading generator from: %s", args.generator_file)
        # style_gan.gen.load_state_dict(torch.load(args.generator_file))
        # Load fewer layers of pre-trained models if possible
        load(style_gan.gen, args.generator_file)
    else:
        logger.info("Training from scratch...")

    if args.discriminator_file is not None:
        logger.info("Loading discriminator from: %s", args.discriminator_file)
        style_gan.dis.load_state_dict(torch.load(args.discriminator_file))

    if args.gen_shadow_file is not None and use_ema:
        logger.info("Loading shadow generator from: %s", args.gen_shadow_file)
        # style_gan.gen_shadow.load_state_dict(torch.load(args.gen_shadow_file))
        # Load fewer layers of pre-trained models if possible
        load(style_gan.gen_shadow, args.gen_shadow_file)

    if args.gen_optim_file is not None:
        logger.info("Loading generator optimizer from: %s", args.gen_optim_file)
        style_gan.gen_optim.load_state_dict(torch.load(args.gen_optim_file))

    if args.dis_optim_file is not None:
        logger.info("Loading discriminator optimizer from: %s", args.dis_optim_file)
        style_gan.dis_optim.load_state_dict(torch.load(args.dis_optim_file))

    # train the network
    style_gan.train(dataset=dataset,
                    num_workers=4,
                    epochs=[2,4,8,8,16,24,32],
                    batch_sizes=[128, 128, 128, 64, 32, 16, 8],
                    fade_in_percentage=[50, 50, 50, 50, 50, 50, 50],
                    logger=logger,
                    output=output_dir,
                    num_samples=36,
                    start_depth= 0,
                    feedback_factor=10,
                    checkpoint_factor=10)
