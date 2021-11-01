"""
gan.py

The file containing the GAN class and its parameters.

Requirements:
    - tensorflow-gpu - 2.4.1
    - matplotlib - 3.4.3

Author: Bobby Melhem
Python Version: 3.9.7
"""


import os
from time import time
from math import log2
from datetime import timedelta
from random import sample, random

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.backend import *

from generator import Generator
from discriminator import Discriminator


#Value for avoiding zero division error
DELTA = 0.000001

#Number of epochs
EPOCHS = 500

#Number of times to run discriminator/generator in seperate training
DISCRIMINATOR_RUNS = 1
GENERATOR_RUNS = 1

#Context specific image settings
IMAGE_SIZE = 128
CHANNELS = 1
BATCH_SIZE = 8 

#Hyper Parameters
LEARNING_RATE = 0.0001
LATENT_SIZE = 256
PENALTY = 10
MIXED_PROBABILITY = 0.9
PATH_LENGTH_DECAY = 0.01
BLOCKS = int(log2(IMAGE_SIZE) - 1)

#Paths for storing progress
CHECKPOINTS_PATH = './checkpoints'
SAMPLE_PATH = './samples'
CACHE_PATH = './cache/'
SAMPLE_COUNT = BATCH_SIZE


class GAN():
    """
    An instance of the StyleGAN.

    Attributes:
        image_size: the size to resize and generate images of form (image_size x image_size).
        learning_rate: the learning rate used in the generator/discriminator optimizers.
        dataset: the tensorflow dataset.
        generator: the generator model of the GAN.
        discriminator: the discriminator model of the GAN.
        generator_loss_averages: Averages of generator losses used for plotting progress of models.
        discriminator_loss_averages: Averages of discriminator losses used for plotting progress of models.
        penalty_factor: factor to scale path length penalty by.
        path_length_average: average path length of generator.
    """


    def __init__(self, image_size, learning_rate):
        """Initialise an instance of the StyleGAN"""

        self.image_size = image_size
        self.learning_rate = learning_rate

        self.dataset = None

        self.generator = Generator(self.image_size, BLOCKS, self.learning_rate, CHANNELS)
        self.discriminator = Discriminator(self.image_size, BLOCKS, self.learning_rate, CHANNELS)

        #Build the models
        self.generator.build_model()
        self.discriminator.build_model()

        self.generator_loss_averages = []
        self.discriminator_loss_averages = []

        self.penalty_factor = PENALTY
        self.path_length_average = 0


    def load_data(self, cache_path, image_paths): 
        """
        Loads the data into a tensorflow tensor and scales down to image size and appropriate channels.
        Segments and caches data into batches.

        Args:
            image_paths : A list of paths to the images to load

        """

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        image_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(image_tensor)
        dataset = dataset.map(preprocessing, num_parallel_calls=8).cache(cache_path)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).batch(BATCH_SIZE)
        self.dataset = dataset


    def generate_style(self, probability):
        """
        Generate latent vector representing style with style mixing based on a probability.

        Args:
            probability : probability of selecting style
        """

        style = [tf.random.normal([BATCH_SIZE, LATENT_SIZE], 0, 1)]

        if random() < probability:
            seed = int(random() * BLOCKS)
            style = (seed * style) + [] + ((BLOCKS - seed) * style)
        else:
            style = BLOCKS * style

        return style


    def gradient_penalty(self, real_batch, fake_batch):
        """
        Calculate the L2 norm of the gradients between real and fake batches.

        Args:
            real_batch : batch of real images
            fake_batch : batch of fake images
                
        Returns:
            The gradient penalty scaled by a factor.
        """

        alpha = tf.random.normal([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
        interpolated = real_batch + alpha * (fake_batch - real_batch)

        with tf.GradientTape() as penalty_tape:

            penalty_tape.watch(interpolated)
            prediction = self.discriminator.model(interpolated, training=True)

        gradients = penalty_tape.gradient(prediction, [interpolated])[0] + DELTA

        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        penalty = tf.reduce_mean((norm - 1.0) ** 2)

        return penalty * self.penalty_factor


    def path_length_regularization(self, w, noise, original_images):
        """
        Penalty which encourages a fixed-sized step in w, to map to a fixed magnitude
        change in the image. It does this by comparing a slightly shifted w batch of 
        generated images and comparing them to the original images.

        Args:
            w : the current w vector
            noise : the input noise
            original_images : batch to compare shift with

        Returns:
            The difference in gradient.
        """

        w_delta = []

        for i in range(len(w)):
            deviation = std(w[i], axis=0, keepdims=True)
            w_delta.append(w[i] + random_normal(tf.shape(w[i])) * deviation * PENALTY)

        new_images = self.generator.model(w_delta + noise)

        gradient_delta = mean(square(new_images - original_images), axis=[1,2,3])

        return gradient_delta


    def train_step_seperated(self, batch, w, penalty): 
        """
        Modified version of the train step where a ratio of how much the discriminator
        and generator run in each batch. This is mostly used when training with a Wasserstein loss.
        (Code slightly deprecated but left in as evidence of attempts at training)

        Args:
            batch : batch of images for the current training step
            w : the w vector used for input
            penalty : boolean indicating whether to apply penalty_factor

        Returns:
            Tuple of averages of the generator and discriminator losses
        """

        discriminator_losses = []
        generator_losses = []

        for round in range(DISCRIMINATOR_RUNS):

            with tf.GradientTape() as discriminator_tape:

                noise = [tf.random.uniform([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], 0, 1)]

                fake_batch = self.generator.model(w + noise)

                real_predictions = self.discriminator.model(batch, training=True)
                fake_predictions = self.discriminator.model(fake_batch, training=True)

                discriminator_loss = self.discriminator.w_loss(real_predictions, fake_predictions)

                discriminator_loss += self.gradient_penalty(batch, fake_batch)

                discriminator_losses.append(discriminator_loss)

                discriminator_variables = self.discriminator.model.trainable_variables

                self.discriminator.optimizer.apply_gradients(zip(discriminator_tape.gradient(discriminator_loss, discriminator_variables), discriminator_variables))

        for round in range(GENERATOR_RUNS):

            with tf.GradientTape() as generator_tape:

                noise = [tf.random.uniform([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], 0, 1)]

                fake_batch = self.generator.model(w + noise)

                fake_predictions = self.discriminator.model(fake_batch, training=True)

                generator_loss = self.generator.w_loss(fake_predictions)

                if penalty:
                    path_lengths = self.path_length_regularization(w, noise, fake_batch)

                    if self.path_length_average > 0:
                        generator_loss += mean(square(path_lengths - self.path_length_average))
                else:
                    path_lengths = self.path_length_average

                if self.path_length_average == 0:
                    self.path_length_average = tf.get_static_value(tf.reduce_mean(path_lengths))

                self.path_length_average = PATH_LENGTH_DECAY * tf.get_static_value(tf.reduce_mean(path_lengths)) + (1 - PATH_LENGTH_DECAY) * self.path_length_average

                generator_losses.append(generator_loss)

                generator_variables = self.generator.model.trainable_variables

                self.generator.optimizer.apply_gradients(zip(generator_tape.gradient(generator_loss, generator_variables), generator_variables))

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_losses)


    def train_step(self, batch, path_regularization): 
        """
        Training step for the StyleGAN over a batch of images.

        Args:
            batch : batch of input images
            path_regularization : whether to calculate new path lengths

        Returns:
            A tuple containing the calculated generator and discriminator loss as well as path lengths.
        """

        style = self.generate_style(MIXED_PROBABILITY)

        noise = [tf.random.uniform([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], 0, 1)]

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            w = [self.generator.style(s) for s in style]

            fake_batch = self.generator.model(w + noise)

            real_predictions = self.discriminator.model(batch, training=True)
            fake_predictions = self.discriminator.model(fake_batch, training=True)

            generator_loss = self.generator.h_loss(fake_predictions)
            discriminator_loss = self.discriminator.h_loss(real_predictions, fake_predictions)

            discriminator_loss += self.gradient_penalty(batch, fake_batch)

            if path_regularization:

                path_lengths = self.path_length_regularization(w, noise, fake_batch)

                if self.path_length_average > 0:
                    generator_loss += mean(square(path_lengths - self.path_length_average))
            else:
                path_lengths = self.path_length_average

            generator_variables = self.generator.model.trainable_variables
            discriminator_variables = self.discriminator.model.trainable_variables

            self.generator.optimizer.apply_gradients(zip(generator_tape.gradient(generator_loss, generator_variables), generator_variables))
            self.discriminator.optimizer.apply_gradients(zip(discriminator_tape.gradient(discriminator_loss, discriminator_variables), discriminator_variables))

        return generator_loss, discriminator_loss, path_lengths


    def train(self, epochs):
        """
        Train the model for the given epochs.

        Args: 
            epochs : The number of epochs to run training for.
        """

        train_start = time()

        for epoch in range(1, epochs + 1):

            epoch_start = time()

            generator_losses = []
            discriminator_losses = []

            fixed_style =  [tf.random.normal([SAMPLE_COUNT, LATENT_SIZE], 0, 1)] * BLOCKS

            fixed_noise = [tf.random.uniform([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], 0, 1)]

            path_regularization = True

            for batch in self.dataset:

                path_regularization = not path_regularization

                generator_loss, discriminator_loss, path_lengths = self.train_step(batch, path_regularization)

                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)

                if self.path_length_average == 0:

                    self.path_length_average = tf.get_static_value(tf.reduce_mean(path_lengths))

                self.path_length_average = PATH_LENGTH_DECAY * tf.get_static_value(tf.reduce_mean(path_lengths)) + (1 - PATH_LENGTH_DECAY) * self.path_length_average
            

            epoch_end = time()
            epoch_time = epoch_end - epoch_start

            self.generator_loss_averages.append(tf.get_static_value(tf.reduce_mean(generator_losses)))
            self.discriminator_loss_averages.append(tf.get_static_value(tf.reduce_mean(discriminator_losses)))

            print("Time taken for epoch" , str(epoch) , ":", str(timedelta(seconds=epoch_time)), \
                ", generator_loss =" , str(self.generator_loss_averages[-1]), \
                ", discriminator_loss =" , str(self.discriminator_loss_averages[-1]))

            self.save_samples(fixed_style, fixed_noise, SAMPLE_PATH, epoch)

            self.plot_losses(SAMPLE_PATH)

            self.save_weights(CHECKPOINTS_PATH, epoch)

        train_end = time()
        train_time = train_end - train_start

        print("Training Time :", str(timedelta(seconds=train_time)))


    def save_weights(self, path, epoch):
        """
        Save the current weights of the model.

        Args:
            path : the path to save the weight files at.
            epoch : the current epoch of training
        """

        if not os.path.exists(path):
            os.mkdir(path)

        self.generator.model.save_weights(path + '/generator/epoch' + str(epoch) + '.ckpt')
        self.discriminator.model.save_weights(path + '/discriminator/epoch' + str(epoch) + '.ckpt')
        self.generator.style.save_weights(path + '/style/epoch' + str(epoch) + '.ckpt')


    def plot_losses(self, path):
        """
        Plot the losses and save them to a file.

        Args: 
            path : the path to save the file to.
        """
        
        plt.clf()
        plt.plot(self.discriminator_loss_averages, label="Discriminator Loss")
        plt.plot(self.generator_loss_averages, label="Generator Loss")
        plt.legend()
        plt.title("StyleGAN2 Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        
        if not os.path.exists(path + '/losses'):
            os.mkdir(path + '/losses')

        plt.savefig(path + '/losses/losses.png')


    def load_weights(self, path):
        """
        Load the model with a file of saved weights at the latest checkpoint.

        Args:
            path : the path to load the weights from.
        """

        self.generator.model.load_weights(tf.train.latest_checkpoint(path + '/generator'))
        self.discriminator.model.load_weights(tf.train.latest_checkpoint(path + '/discriminator'))
        self.generator.style.load_weights(tf.train.latest_checkpoint(path + '/style'))


    def save_samples(self, style, noise, path, epoch):
        """
        Save a batch of images to a file based on the given epoch.

        Args:
            style : style to generate images from
            noise : noise to generate images from
            path : the path to save the samples to
            epoch : the current epoch
        """

        samples = self.generator.model(style + noise)

        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(path + '/epoch' + str(epoch)):
            os.mkdir(path + '/epoch' + str(epoch))

        for index in range(len(samples)):
            tf.keras.preprocessing.image.save_img(path + '/epoch' + str(epoch) + '/' + str(index) + '.png', samples[index])


def preprocessing(path):
    """
    Preprocessing function for loading images.

    Args:
        path : the path to the image to process

    Returns:
        The formatted image.
    """

    #Make grayscale
    image = tf.image.decode_png(tf.io.read_file(path), channels=CHANNELS)
    #Resize to image size
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    #Normalise
    image = tf.cast(image, tf.float32) / 255.0

    return image

