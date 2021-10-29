import os
import datetime
import time
import timeit

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