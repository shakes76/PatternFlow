import os

import tensorflow as tf
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython import display


class DCGAN:

    def __init__(self, img_size, input_shape=256, batch_size=256, noise_dim=256):
        """
        A DCGAN object that can create DCGAN structure and train. The object is flexible to different output size
        :param img_size: output size
        :param input_shape: The input shape usually equals to noise_dim
        :param batch_size: The batch size for training, default is 256
        :param noise_dim: the noise dimension. Default is 256
        """
        # Parameter initialize
        self.img_size = img_size
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.noise_dim = noise_dim
        # Model initialize
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # Checkpoint initialize
        self.checkpoint_dir = '../models/training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def generator_model(self):
        """
        A function that constructs the generator model
        :return: A generator follows DCGAN standard
        """
        # The Scale of up sampling is defined as 1/8, 1/4, 1/2, 1 of the output image size
        starter = int(self.img_size / 8)

        model = tf.keras.Sequential()
        model.add(layers.Dense(starter * starter * self.batch_size, use_bias=False, input_shape=(self.input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Reshape((starter, starter, self.batch_size)))
        assert model.output_shape == (None, starter, starter, self.batch_size)

        model.add(layers.Conv2DTranspose(128, (6, 6), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, starter, starter, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 2 * starter, 2 * starter, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(32, (6, 6), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 4 * starter, 4 * starter, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(1, (6, 6), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.img_size, self.img_size, 1)
        return model

    def discriminator_model(self):
        """
        A function that constructs the discriminator model
        :return: A discriminator follows DCGAN standard
        """
        model = tf.keras.Sequential()
        model.add(
            layers.Conv2D(32, (6, 6), strides=(2, 2), padding='same', input_shape=[self.img_size, self.img_size, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        """
        A function that takes the output from generator and real image and returns the loss
        :param real_output: The output from discriminator of real image
        :param fake_output: The output from discriminator of fake image
        :return: Total loss between real img and fake img
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        """
        A function that takes the output from generator and returns the loss
        :param fake_output: The output from generator
        :return: The cross entropy loss of generator
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        """
        A child function that handles the training part for each batch
        :param images: The real image
        :return:
        """
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def generate_and_save_images(self, epoch, test_input):
        """
        A function that renders and saves the image made by generator
        :param epoch: The epoch for the training
        :param test_input: The output from generator
        :return: Nothing
        """
        predictions = self.generator(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(np.array((predictions[i, :, :, 0] * 127.5 + 127.5)).astype(np.uint8), cmap='gray')
            plt.axis('off')
        if (epoch + 1) % 50 == 0:
            plt.savefig('../images/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def train(self, dataset, epochs):
        """
        A function that handles main part of training.
        :param dataset: The training dataset
        :param epochs: The epoch of training
        :return: nothing
        """

        num_examples_to_generate = 16
        seed = tf.random.normal([num_examples_to_generate, self.noise_dim])
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            display.clear_output(wait=True)
            self.generate_and_save_images(epoch + 1, seed)

            # Save the model every 500 epoch
            if (epoch + 1) % 500 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        display.clear_output(wait=True)
        self.generate_and_save_images(epochs, seed)

    def load_model(self):
        """
        Load model from latest check point
        :return:
        """
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def generate_images(self, num=1):
        """

        :param num: The number of fake images needs to generate
        :return: A series of fake images
        """
        noise = tf.random.normal([num, self.noise_dim])
        generated_image = self.generator(noise, training=False)
        plt.imshow(np.array((generated_image[0, :, :, 0] * 127.5 + 127.5)).astype(np.uint8), cmap='gray')
        return generated_image
