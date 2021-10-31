from dataclasses import dataclass
import torch
from torch import optim
from torchvision import datasets

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

        # "We implement progressive grow ing the same way as Karras et al. [30], but we start from 8^2
        # images instead of 4^2"
        self.state.progressive_level = 1

        # discriminator is trained only at every critic_iteration iterations
        # We first train discriminator discriminator_iteration cycles, then one generator, and keep repeating that
        # discriminator needs a little bit more training so it is not overpowered by the generator [Udemy GANs course]
        #TODO do this progressively too?
        self.generator_iteration = 5

        # maximum number of progressive levels
        self.max_progressive_level = 7
        self.start_image_resolution = 4  # we define starting level using state.progressive_level
        self.input_dim = 4

        # values provided in the paper
        # "and the dimensionality of all input and output activations — including z and w — is 512."
        self.z_dim = 512

        self.number_of_mapping_network_fc_layers = 8

        # input - images folder
        self.input_path = './images/'
        # output - trained models folder
        self.models_path = './models/'
        # save checkpoint/model
        self.continue_from_previous_checkpoint = True

        # the number of training samples
        # ... and training length of 150,000 images." [page 9]
        self.images_per_progress_level = 150000  # for progressing to the next level
        # "we thus increase the training time from 12M to 25M images" [page 9]
        self.training_time = 25000000  # total for training

        self.gen_losses = []
        self.disc_losses = []
        self.show_loss_steps = 10

        # "we found that setting the learning rate to 0.002 instead of 0.003 for 512^2 and 1024^2 leads to
        # better stability" [page 9]
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

        # Optimizers
        # "We use the learning rate of 10−3, minibatch size of 8, Adam optimizer, and ..."
        self.gen_optim = optim.Adam([{
            'params': self.generator.synthesis_network.parameters(),
            'lr': 0.001
        }, {
            'params': self.generator.to_rgbs.parameters(),
            'lr': 0.001
        }], lr=0.001, betas=(0.0, 0.99))
        self.gen_optim.add_param_group({
            'params': self.generator.mapping_network.parameters(),
            'lr': 0.001 * 0.01,
            'mul': 0.01
        })
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))
        self.dataset = datasets.ImageFolder(self.input_path)


    def get_images(self, dataset, batch_size, image_size):
        pass

    def save_generated_image(self, tensor, i):
        # TODO
        pass

    def train(self):

        # Training loop
        while self.state.used_sample < self.training_time:
            self.state.iteration += 1

            # -------- Update discriminator, maximise log(D(x)) + log(1 - D(G(z))) ---------
            # TODO

            if self.state.iteration % self.discriminator_iteration == 0:
                # update generator after training discriminator_cycles of discriminator

                # ------ Update Generator, maximise log(D(G(z))) --------
                # TODO

                # TODO make a progress bar
                pass

        # TODO save final model
        print(f'Final model saved.')

        return

    def fit(self, continue_from_previous_checkpoint=False, start_point=0):
        # if continue_from_previous_checkpoint:
        #     self.load_model(start_point)

        self.state.startpoint = self.state.used_sample

        self.generator.train()
        self.discriminator.train()
        disc_losses, gen_losses = self.train()
