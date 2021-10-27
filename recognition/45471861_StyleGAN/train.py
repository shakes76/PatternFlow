# !/user/bin/env python
"""
The module controls the StyleGAN training
"""
import os.path

import numpy as np
import tensorflow as tf
from time import time
from models import Generator, Discriminator
import matplotlib.pyplot as plt
import neptune.new as neptune
from neptune.new.types import File
from datetime import datetime
from tensorflow.keras.utils import image_dataset_from_directory

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


class Trainer:
    def __init__(self, data_folder: str, output_dir: str, g_input_res, g_init_filters, d_final_res, d_input_filters,
                 width=64, height=64, channels=1, latent_dim=100, batch=128, epochs=20, checkpoint=1, lr=0.0002,
                 beta_1=0.5, validation_images=16, seed=1, use_neptune=False):

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
        self.generator = Generator(lr, beta_1, latent_dim, g_input_res, width, g_init_filters)
        self.generator.build()
        self.discriminator = Discriminator(lr, beta_1, width, d_final_res, d_input_filters)
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
        self.validation_latent_single = tf.random.normal([1, latent_dim], seed=seed)

        # credential for neptune
        self.neptune = use_neptune
        self.run = None
        if self.neptune:
            with open("neptune_credential.txt", 'r') as credential:
                token = credential.readline()

            self.run = neptune.init(
                project="zhien.zhang/styleGAN",
                api_token=token,
            )

            self.run["Image resolution"] = width
            self.run["Epochs"] = epochs
            self.run["Batch size"] = self.batch
            self.run["Latent dim"] = self.latent_dim
            self.run["G input resolution"] = g_input_res
            self.run["G initial filters"] = g_init_filters
            self.run["D input filters"] = d_input_filters
            self.run["D final resolution"] = d_final_res

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

    def _train_g(self, fade_in):
        latent = tf.random.normal([self.batch, self.latent_dim])

        with tf.GradientTape() as tape:
            fake = self.generator.model(latent, training=True)
            fake_score = self.discriminator.model((fake, fade_in), training=False)
            loss = self.generator.loss(fake_score)

        gradient = tape.gradient(loss, self.generator.model.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradient, self.generator.model.trainable_variables))

        score = tf.reduce_mean(fake_score)
        return score

    def _train_d(self, real, fade_in):
        latent = tf.random.normal([self.batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator.model(latent, training=False)
            fake_score = self.discriminator.model((fake, fade_in), training=True)
            real_score = self.discriminator.model((real, fade_in), training=True)
            loss = self.discriminator.loss(real_score, fake_score)

        gradient = tape.gradient(loss, self.discriminator.model.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradient, self.discriminator.model.trainable_variables))

        score = 1/2 * tf.reduce_mean(real_score) + 1/2 * tf.reduce_mean(1 - fake_score)
        return score

    def _show_images(self, epoch, save=True):
        predictions = self.generator.model(self.validation_latent, training=False)

        fig = plt.figure(figsize=(7, 7))

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
            fade_in = epoch / float(self.epochs - 1)

            for image_batch in self.dataset:
                # normalize to the range [-1, 1] to match the generator output
                image_batch = (image_batch - 255 / 2) / (255 / 2)

                d_score = self._train_d(image_batch, fade_in)
                g_score = self._train_g(fade_in)

                # log to neptune
                if self.neptune:
                    self.run["G_loss"].log(g_score)
                    self.run["D_loss"].log(d_score)

                iter += 1

                # showing the result every 100 iterations
                if iter % 100 == 0:
                    fig = self._show_images(0, save=False)
                    if self.neptune:
                        self.run["Validation"].upload(fig)

            # show and save the result
            if epoch % self.checkpoint == 0:
                fig = self._show_images(epoch, save=True)

                if self.neptune:
                    self.run["Train/epoch_{}".format(epoch)].upload(fig)
                    single_image = self.generator.model(self.validation_latent_single, training=False)
                    single_image = tf.reshape(single_image, (self.width, self.height, self.channels))
                    # normalize to [0, 1]
                    single_image_norm = single_image * 0.5 + 0.5
                    single_image_norm = np.clip(single_image_norm, 0, 1)
                    single_image_norm = File.as_image(single_image_norm)
                    self.run["Train/single"].log(single_image_norm)  # save the raw array
                    # normalize to [0, 255]
                    fig = plt.figure(figsize=(7, 7))
                    plt.imshow(single_image * 127.5 + 127.5, cmap='gray')
                    plt.axis('off')
                    self.run["Train/epoch_{}_single".format(epoch)].upload(fig)

                print('Time for epoch {} is {} sec'.format(epoch + 1, time() - start))
                print("Discriminator score: {}\t Generator score: {}".format(d_score, g_score))

        # save D and G
        folder = os.path.join(self.output_dir, "Model")
        g_folder = os.path.join(folder, "G")
        d_folder = os.path.join(folder, "D")
        self.generator.model.save(g_folder)
        self.discriminator.model.save(d_folder)
