
import torch

from tk_model import StyleBasedGenerator, Discriminator

device = torch.device('cuda:0')

# We first train discriminator discriminator_cycles times (cycles), then one of the generator,
# then again discriminator_cycles of the discriminator, then one generator
# discriminator needs a little bit more training so it is not overpowered by the generator
DISCRIMINATOR_CYCLES = 1

MAX_PROGRESSIVE_LEVEL = 7  # the maximum number of progressive levels
START_IMAGE_RESOLUTION = 4

# input
image_folder_path = './images/'

# save checkpoint/model
save_folder_path = './models/'
continue_from_previous_checkpoint = True

# model state to save
model_state = {
    "progressive_level": 0,
    "iteration": 0,
    "startpoint": 0,
    "used_sample": 0,
    "alpha": 0
}

# values provided in the paper
z_dim = 512
number_of_fc_layers = 8

# the number of samples
# TODO maybe make a map for progress_level_samples
progress_level_samples = 1000  # for progressing to the next level
total_samples = 100000  # for training

input_dim = 4
gen_losses = []
disc_losses = []

batch_size = {4: 256, 8: 128, 16: 64, 32: 32, 64: 16, 128: 8}
mini_batch_size = 8

# Get sample
def get_images(dataset, batch_size, image_size):
    print("Loading image batch...")
    pass


def save_generated_image(tensor, i):
    pass


def train(dataset, generator, gen_optim, gen_losses, discriminator, disc_optim, disc_losses, model_state):

    # progressive training - setting resolution for current level
    res = START_IMAGE_RESOLUTION * (2 ** model_state["progressive_level"])

    # load next batch of images for current image resolution

    # Training loop
    while model_state["used_sample"] < total_samples:
        model_state["iteration"] += 1

        # -------- Update discriminator, maximise log(D(x)) + log(1 - D(G(z))) ---------
        # TODO

        if model_state["iteration"] % DISCRIMINATOR_CYCLES != 0: continue
        # update generator after training discriminator_cycles of discriminator

        # ------ Update Generator, maximise log(D(G(z))) --------
        # TODO

        # TODO make a progress bar

    # TODO save final model
    print(f'Final model saved.')

    return


if __name__ == '__main__':
    # Create models
    generator = StyleBasedGenerator(number_of_fc_layers, z_dim, input_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    gen_optim = None
    disc_optim = None

    # Dataset
    dataset = None

    generator.train()
    discriminator.train()

    disc_losses, gen_losses = train(dataset,
                                    generator, gen_optim, gen_losses,
                                    discriminator, disc_optim, disc_losses,
                                    model_state)



