import os
from pathlib import Path

from dataclasses import dataclass, asdict
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from tk_model import Discriminator
from tk_model import StyleBasedGenerator
from training_arguments import TrainingArguments


@dataclass
class ModelState:
    progressive_level: int = 0
    current_iteration: int = 0
    start_sample_point: int = 0
    used_samples: int = 0
    alpha: int = 0


class StyleGanTrainer:

    def __init__(self, training_arguments:TrainingArguments):
        self.device = torch.device(training_arguments.device)

        self.state = ModelState()

        # "We implement progressive grow ing the same way as Karras et al. [30], but we start from 8^2
        # images instead of 4^2"
        self.state.progressive_level = training_arguments.start_progressive_level

        # discriminator is trained only at every critic_iteration iterations
        # We first train discriminator discriminator_iteration cycles, then one generator, and keep repeating that
        # discriminator needs a little bit more training so it is not overpowered by the generator [Udemy GANs course]
        #TODO do this progressively too?
        self.generator_iteration = 5

        # maximum number of progressive levels
        self.max_progressive_level = training_arguments.max_progressive_level
        self.start_image_resolution = 4  # we define starting level using state.progressive_level
        self.input_dim = 4

        # values provided in the paper
        # "and the dimensionality of all input and output activations — including z and w — is 512."
        self.z_dim = 512

        self.number_of_mapping_network_fc_layers = 8

        # input - images folder
        self.input_path = 'images/'
        # output - trained models folder
        self.models_path = 'trained_models/'
        # save checkpoint/model
        self.continue_from_previous_checkpoint = True
        self.save_steps = 1000

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

    def require_grad_flag(self, module, flag):
        '''
        Setting the flag for gradient computation
        '''
        for parameter in module.parameters():
            parameter.requires_grad = flag

    def reset_learning_rate(self, optimizer, lr):
        '''
        Resetting the learning rate for optimizer
        '''
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg.get('mul', 1)

    def get_images(self, dataset, batch_size, image_size):
        '''
        Getting next batch of images
        '''
        print(f"Loading image batch, batch_size= {batch_size}")
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset.transform = transform
        torch.cuda.empty_cache()
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def save_generated_image(self, tensor, i):
        '''
        Saving generated image
        '''
        grid = tensor[0].clamp_(-1, 1).add_(1).div_(2)
        # de-normalise and with pytorch we want the opposite order of the channels ie we have 128 x 128 x 3,
        # but we want 3 x 128 x 128 (permute)
        img_matrix = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        Image.fromarray(img_matrix).save(f'previews/generated-{i}.png')

    def update_state(self):
        self.state.progressive_level += 1
        self.state.alpha = 0
        self.state.start_sample_point = self.state.used_samples
        print(f"Current state: {self.state}")

    def update_learning_rates(self, current_learning_rate):
        self.reset_learning_rate(self.gen_optim, current_learning_rate)
        self.reset_learning_rate(self.disc_optim, current_learning_rate)

    def execute_generator_step(self, w2, gen_noise, alpha):
        self.require_grad_flag(self.generator, False)

        self.generator.zero_grad()
        self.require_grad_flag(self.discriminator, False)
        self.require_grad_flag(self.generator, True)

        fake_image = self.generator(w2, self.state.progressive_level, alpha, gen_noise)
        fake_logit = self.discriminator(fake_image, self.state.progressive_level, alpha)

        fake_loss = nn.functional.softplus(-fake_logit).mean()
        fake_loss.backward()
        self.gen_optim.step()

        self.gen_losses.append(fake_loss.item())

        return fake_image

    def execute_discriminator_real_step(self, real_image, alpha):
        # Train Discriminator discriminator_cycles times, then one Generator
        self.discriminator.zero_grad()
        self.require_grad_flag(self.discriminator, True)

        # Real image predict & backward
        # We only implement non-saturating loss with R1 regularization loss
        real_image.requires_grad = True

        real_logit = self.discriminator(real_image, self.state.progressive_level, alpha)

        real_loss = nn.functional.softplus(-real_logit).mean()
        # real_loss.backward(retain_graph=True)

        # "... loss [22] with R1 regularization [44] using γ = 10"
        grad_real = torch.autograd.grad(outputs=real_logit.sum(), inputs=real_image, create_graph=True)[0]
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        # grad_penalty_real.backward()

        d_loss = real_loss + grad_penalty_real
        d_loss.backward(retain_graph=True)

        return real_loss

    def execute_discriminator_fake_step(self, w1, disc_noise, alpha):
        fake_image = self.generator(w1, self.state.progressive_level, alpha, disc_noise)
        fake_logit = self.discriminator(fake_image, self.state.progressive_level, alpha)

        fake_loss = nn.functional.softplus(fake_logit).mean()
        fake_loss.backward()
        self.disc_optim.step()

        return fake_loss

    def generate_noise(self, current_batch_size):
        disc_noise = []
        gen_noise = []
        for level in range(self.state.progressive_level + 1):
            # the size of noise vector grows depends on level, when we do upsampling
            size = 4 * (2 ** level)
            disc_noise.append(torch.randn((current_batch_size, 1, size, size), device=self.device))
            gen_noise.append(torch.randn((current_batch_size, 1, size, size), device=self.device))

        return disc_noise, gen_noise

    def train(self):
        # progressive training - setting resolution for current level
        image_resolution = self.start_image_resolution * (2 ** self.state.progressive_level)
        current_batch_size = self.batch_size.get(image_resolution, self.default_batch_size)
        current_learning_rate = self.learning_rate.get(image_resolution, self.default_learning_rate)

        self.update_learning_rates(current_learning_rate)

        # load next batch of images for current image resolution
        image_loader = self.get_images(self.dataset, current_batch_size, image_resolution)
        data_loader = iter(image_loader)

        progress_bar = tqdm(total=self.training_time, initial=self.state.used_samples)

        # Training loop
        while self.state.used_samples < self.training_time:
            alpha = min(1, self.state.alpha + current_batch_size / self.images_per_progress_level)
            self.state.current_iteration += 1

            # check if we need to advance to the next progressive level
            if (self.state.used_samples - self.state.start_sample_point) > self.images_per_progress_level and \
                    self.state.progressive_level < self.max_progressive_level:

                self.update_state()

                image_resolution = self.start_image_resolution * (2 ** self.state.progressive_level)
                print(f"Now training for resolution {image_resolution} x {image_resolution}")
                current_batch_size = self.batch_size.get(image_resolution, self.default_batch_size)
                current_learning_rate = self.learning_rate.get(image_resolution, self.default_learning_rate)
                self.update_learning_rates(current_learning_rate)

                image_loader = self.get_images(self.dataset, current_batch_size, image_resolution)
                data_loader = iter(image_loader)

            try:
                real_image, _ = next(data_loader)
            except (OSError, StopIteration):
                data_loader = iter(image_loader)
                real_image, _ = next(data_loader)

            self.state.used_samples += real_image.shape[0]
            progress_bar.update(real_image.shape[0])

            real_image = real_image.to(self.device)

            w1 = [torch.randn((current_batch_size, self.z_dim), device=self.device)]
            w2 = [torch.randn((current_batch_size, self.z_dim), device=self.device)]

            ### DISCRIMINATOR
            real_loss = self.execute_discriminator_real_step(real_image, alpha)
            disc_noise, gen_noise = self.generate_noise(current_batch_size)
            fake_loss = self.execute_discriminator_fake_step(w1, disc_noise, alpha)

            self.disc_losses.append((real_loss + fake_loss).item())

            ### GENERATOR
            if self.state.current_iteration % self.generator_iteration == 0:
                # train generator after generator_iteration of discriminator cycles
                fake_image = self.execute_generator_step(w2, gen_noise, alpha)

                if self.state.current_iteration % (self.show_loss_steps * 100) == 0:
                    self.save_generated_image(fake_image.data.cpu(), self.state.current_iteration)

            if self.state.current_iteration % self.save_steps == 0:
                # save checkpoint
                self.save_model(f'{self.models_path}/checkpoints/model{self.state.current_iteration}.pth')
                print(f'Model checkpoint {self.state.current_iteration} saved.')

            if len(self.disc_losses) > 0 and len(self.gen_losses) > 0:
                progress_bar.set_description((
                    f'Resolution: {image_resolution}*{image_resolution}, disc_loss: {self.disc_losses[-1]:.5f}, '
                    f'gen_loss: {self.gen_losses[-1]:.5f}, alpha: {alpha:.4f}'))

        # save final model
        self.save_model(f'{self.models_path}/model.pth')
        print(f'Final model saved.')

        return self.disc_losses, self.gen_losses

    def fit(self, continue_from_previous_checkpoint=False, start_point=0):
        if continue_from_previous_checkpoint:
            self.load_model(start_point)

        self.state.start_sample_point = self.state.used_samples

        self.generator.train()
        self.discriminator.train()
        disc_losses, gen_losses = self.train()

    def save_model(self, save_path):
        # create save directory if it does not exist
        Path(f"{self.models_path}/checkpoints").mkdir(parents=True, exist_ok=True)

        torch.save({
            'generator': self.generator.state_dict(),
            'gen_optim': self.gen_optim.state_dict(),
            'gen_losses': self.gen_losses,

            'discriminator': self.discriminator.state_dict(),
            'disc_optim': self.disc_optim.state_dict(),
            'disc_losses': self.disc_losses,

            'model_state': self.state,
        }, save_path)
        print(f'Model saved to {save_path}.')

    def load_model(self, iteration):
        if not os.path.exists(f'{self.models_path}/checkpoints/model{iteration}.pth'):
            print('No pre-trained model found, training from scratch...')
        else:
            # Load data
            print(f'Loading pre-trained model: {self.models_path}/checkpoints/model{iteration}.pth ...')
            checkpoint = torch.load(f'{self.models_path}/checkpoints/model{iteration}.pth')

            self.generator.load_state_dict(checkpoint['generator'])
            self.gen_optim.load_state_dict(checkpoint['gen_optim'])
            self.gen_losses = checkpoint.get('gen_losses', [])

            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.disc_optim.load_state_dict(checkpoint['disc_optim'])
            self.disc_losses = checkpoint.get('disc_losses', [])

            self.state = checkpoint['model_state']
