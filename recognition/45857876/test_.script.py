import os
import datetime
import time
import timeit
import argparse

import torch
from torch.backends import cudnn

from data import make_dataset
from utils import make_logger, list_dir_recursively_with_ignore, copy_files_and_create_dirs
from models.GAN import StyleGAN

def train(self, dataset, num_workers, epochs, batch_sizes, fade_in_percentage, logger, output,
          num_samples=36, start_depth=0, feedback_factor=100, checkpoint_factor=1):

    assert self.depth <= len(epochs), "epochs not compatible with depth"
    assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
    assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

    # turn the generator and discriminator into train mode
    self.gen.train()
    self.dis.train()
    if self.use_ema:
        self.gen_shadow.train()

    # create a global time counter
    global_time = time.time()

    # create fixed_input for debugging
    fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

    # config depend on structure
    logger.info("Starting the training process ... \n")
    if self.structure == 'fixed':
        start_depth = self.depth - 1
    step = 1  # counter for number of iterations
    for current_depth in range(start_depth, self.depth):
        current_res = np.power(2, current_depth + 2)
        logger.info("Currently working on depth: %d", current_depth + 1)
        logger.info("Current resolution: %d x %d" % (current_res, current_res))

        ticker = 1

        # Choose training parameters and configure training ops.
        # TODO
        data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

        for epoch in range(1, epochs[current_depth] + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            logger.info("Epoch: [%d]" % epoch)
            # total_batches = len(iter(data))
            total_batches = len(data)

            fade_point = int((fade_in_percentage[current_depth] / 100)
                             * epochs[current_depth] * total_batches)

            for (i, batch) in enumerate(data, 1):
                # calculate the alpha for fading in the layers
                alpha = ticker / fade_point if ticker <= fade_point else 1

                # extract current batch of data for training
                images = batch.to(self.device)
                gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha)

                # optimize the generator:
                gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha)

                # provide a loss feedback
                if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                    logger.info(
                        "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                        % (elapsed, step, i, dis_loss, gen_loss))

                    # create a grid of samples and save it
                    os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                    gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                + "_" + str(epoch) + "_" + str(i) + ".png")

                    with torch.no_grad():
                        self.create_grid(
                            samples=self.gen(fixed_input, current_depth, alpha).detach() if not self.use_ema
                            else self.gen_shadow(fixed_input, current_depth, alpha).detach(),
                            scale_factor=int(
                                np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                            img_file=gen_img_file,
                        )

                # increment the alpha ticker and the step
                ticker += 1
                step += 1

            elapsed = timeit.default_timer() - start
            elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
            logger.info("Time taken for epoch: %s\n" % elapsed)

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                save_dir = os.path.join(output, 'models')
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(
                    save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(
                    save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")

                torch.save(self.gen.state_dict(), gen_save_file)
                logger.info("Saving the model to: %s\n" % gen_save_file)
                torch.save(self.dis.state_dict(), dis_save_file)
                torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                # also save the shadow generator if use_ema is True
                if self.use_ema:
                    gen_shadow_save_file = os.path.join(
                        save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                    logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

    logger.info('Training completed.\n')

def create_grid(samples, scale_factor, img_file):
    """
    utility function to create a grid of GAN samples

    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate

    # upsample the image
    if scale_factor > 1:
        samples = interpolate(samples, scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
               normalize=True, scale_each=True, pad_value=128, padding=1)


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

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    # make output dir
    output_dir = opt.output_dir
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
    logger = make_logger("project", opt.output_dir, 'log')

    # device
    if opt.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id
        num_gpus = len(opt.device_id.split(','))
        logger.info("Using {} GPUs.".format(num_gpus))
        logger.info("Training on {}.\n".format(torch.cuda.get_device_name(0)))
        cudnn.benchmark = True
    device = torch.device(opt.device)

    # create the dataset for training
    dataset = make_dataset(opt.dataset)

    # init the network
    style_gan = StyleGAN(structure=opt.structure,
                         resolution=opt.dataset.resolution,
                         num_channels=opt.dataset.channels,
                         latent_size=opt.model.gen.latent_size,
                         g_args=opt.model.gen,
                         d_args=opt.model.dis,
                         g_opt_args=opt.model.g_optim,
                         d_opt_args=opt.model.d_optim,
                         loss=opt.loss,
                         drift=opt.drift,
                         d_repeats=opt.d_repeats,
                         use_ema=opt.use_ema,
                         ema_decay=opt.ema_decay,
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

    if args.gen_shadow_file is not None and opt.use_ema:
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
                    num_workers=opt.num_works,
                    epochs=opt.sched.epochs,
                    batch_sizes=opt.sched.batch_sizes,
                    fade_in_percentage=opt.sched.fade_in_percentage,
                    logger=logger,
                    output=output_dir,
                    num_samples=opt.num_samples,
                    start_depth=args.start_depth,
                    feedback_factor=opt.feedback_factor,
                    checkpoint_factor=opt.checkpoint_factor)