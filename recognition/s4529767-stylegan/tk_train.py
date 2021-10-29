from dataclasses import dataclass, asdict
import torch

from tk_model import Discriminator
from tk_model import StyleBasedGenerator


@dataclass
class ModelState:
    progressive_level: int = 0
    iteration: int = 0
    startpoint: int = 0
    used_sample: int = 0
    alpha: int = 0


class StyleGanTrainer:

    def __init__(self, device):

        self.device = torch.device(device)

        self.state = ModelState()
        self.state.progressive_level = 1 # start from 8 x 8, skipping 4 x 4 level

        # We first train discriminator discriminator_cycles times (cycles), then one of the generator,
        # then again discriminator_cycles of the discriminator, then one generator
        # discriminator needs a little bit more training so it is not overpowered by the generator
        #TODO do this progressively too?
        self.critic_iteration = 5

        # maximum number of progressive levels
        self.max_progressive_level = 7
        self.start_image_resolution = 4
        self.input_dim = 4

        # values provided in the paper
        self.z_dim = 512
        self.number_of_mapping_network_fc_layers = 8

        # input - images folder
        self.image_folder_path = './images/'
        # output - trained models folder
        self.save_folder_path = './models/'
        # save checkpoint/model
        self.continue_from_previous_checkpoint = True

        self.total_samples = 100000000  # for training

        self.gen_losses = []
        self.disc_losses = []

        self.learning_rate = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        self.default_learning_rate = 0.001

        # due to GPU memory error, had to use these values on 3080 GPU
        self.batch_size = {4: 128, 8: 128, 16: 64, 32: 32, 64: 8, 128: 4}
        self.default_batch_size = 4

        # Create models
        self.generator = StyleBasedGenerator(self.number_of_mapping_network_fc_layers,
                                        self.z_dim,
                                        self.input_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

    def get_images(self, dataset, batch_size, image_size):
        pass

    def save_generated_image(self, tensor, i):
        # TODO
        pass

    def train(self):

        # Training loop
        while self.state.used_sample < self.total_samples:
            self.state.iteration += 1

            # -------- Update discriminator, maximise log(D(x)) + log(1 - D(G(z))) ---------
            # TODO

            if self.state.iteration % self.critic_iteration == 0:
                # update generator after training discriminator_cycles of discriminator

                # ------ Update Generator, maximise log(D(G(z))) --------
                # TODO

                # TODO make a progress bar
                pass

        # TODO save final model
        print(f'Final model saved.')

        return

    def fit(self, continue_from_previous_checkpoint=False, start_point=0):
        if continue_from_previous_checkpoint:
            self.load_model(start_point)

        self.state.startpoint = self.state.used_sample

        self.generator.train()
        self.discriminator.train()
        disc_losses, gen_losses = self.train()
