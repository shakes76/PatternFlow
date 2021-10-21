# !/user/bin/env python
"""
The module controls the StyleGAN training
"""
import os.path

import tensorflow as tf
from time import time
from models import Generator, Discriminator
import matplotlib.pyplot as plt
import neptune.new as neptune
from datetime import datetime
from tensorflow.keras.utils import image_dataset_from_directory

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


class Trainer:
    def __init__(self, data_folder: str, output_dir: str, width=64, height=64, channels=1, latent_dim=100, batch=128,
                 epochs=20, checkpoint=1, lr=0.0002, beta_1=0.5, validation_images=16, seed=1, neptune=False):
        self.width = width
        self.height = height
        self.channels = channels
        self.rgb = (channels == 3)
        self.latent_dim = latent_dim
        self.batch = batch
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.lr = lr
        self.beta_1 = beta_1
        self.num_of_validation_images = validation_images
        self.output_dir = self._create_output_folder(output_dir)

        # initialize models
        self.generator = Generator(lr, beta_1, latent_dim)
        self.generator.build()
        self.discriminator = Discriminator(lr, beta_1)
        self.discriminator.build()

        # data
        self.dataset = None
        if channels == 1:
            color_mod = "grayscale"
        else:
            color_mod = "rgb"
        self.load_data(data_folder, (width, height), color_mod=color_mod)

        # latent code for validation
        self.validation_latent = tf.random.normal([self.num_of_validation_images, latent_dim], seed=seed)

        # credential for neptune
        self.neptune = neptune
        self.run = None
        if self.neptune:
            with open("neptune_credential.txt", 'r') as credential:
                token = credential.readline()

            self.run = neptune.init(
                project="zhien.zhang/styleGAN",
                api_token=token,
            )

    @staticmethod
    def _create_output_folder(upper_folder: str):
        run_folder = datetime.now().strftime("%d-%m/%Y_%H_%M_%S")
        output_folder = os.path.join(upper_folder, run_folder)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def load_data(self, image_folder, image_size: tuple, color_mod="grayscale"):
        train_batches = image_dataset_from_directory(
            image_folder, labels=None, label_mode=None,
            class_names=None, color_mode=color_mod, batch_size=self.batch, image_size=image_size, shuffle=True,
            seed=None,
            validation_split=None, subset=None,
            interpolation='bilinear', follow_links=False,
            crop_to_aspect_ratio=False
        )
        self.dataset = train_batches

    def _train_g(self):
        latent = tf.random.normal([self.batch, self.latent_dim])

        with tf.GradientTape() as tape:
            fake = self.generator.model(latent, training=True)
            fake_score = self.discriminator.model(fake, training=False)
            loss = self.generator.loss(fake_score)

        gradient = tape.gradient(loss, self.generator.model.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradient, self.generator.model.trainable_variables))

        score = tf.reduce_mean(fake_score)
        return score

    def _train_d(self, real):
        latent = tf.random.normal([self.batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator.model(latent, training=False)
            fake_score = self.discriminator.model(fake, training=True)
            real_score = self.discriminator.model(real, training=True)
            loss = self.discriminator.loss(real_score, fake_score)

        gradient = tape.gradient(loss, self.discriminator.model.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradient, self.discriminator.model.trainable_variables))

        score = 1/2 * tf.reduce_mean(real_score) + 1/2 * tf.reduce_mean(1 - fake_score)
        return score

    def _show_images(self, epoch, save=True):
        predictions = self.generator.model(self.validation_latent, training=False)

        fig = plt.figure(figsize=(5, 5))

        predictions = tf.reshape(predictions, (-1, self.width, self.height, self.channels))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)

            if self.rgb:
                plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
            else:
                plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        if save:
            path = os.path.join(self.output_dir, 'image_at_epoch_{}.png'.format(epoch))
            plt.savefig(path)

        plt.show()

        return fig

    def train(self):
        iter = 0
        for epoch in range(self.epochs):
            start = time()

            for image_batch in self.dataset:
                # normalize to the range [-1, 1] to match the generator output
                image_batch = (image_batch - 255 / 2) / (255 / 2)

                d_score = self._train_d(image_batch)
                g_score = self._train_g()

                # log to neptune
                if self.neptune:
                    self.run["Generator_Score"].log(g_score)
                    self.run["Discriminator_Score"].log(d_score)

                iter += 1

                # showing the result every 100 iterations
                if iter % 100 == 0:
                    fig = self._show_images(0, save=False)
                    if self.neptune:
                        self.run["Validation"].upload(fig)

            # show and save the result
            if epoch % self.checkpoint == 1:
                self._show_images(epoch, save=True)
                print('Time for epoch {} is {} sec'.format(epoch + 1, time() - start))
                print("Discriminator score: {}\t Generator score: {}".format(d_score, g_score))
