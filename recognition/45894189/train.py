import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import load_data
from tensorflow import keras
from modules import Generator, Discriminator
from util import ImageSaver, WeightSaver

"""
train.py defines a StyleGAN model and its training procedure

The following example was used as a guide:
Face image generation with StyleGAN. Soon-Yau CHeong. Date Created: 01/07/2021, 
Last Modified: 20/12/2021. https://keras.io/examples/generative/stylegan/

Author: Sam Smeaton 45894189 21/10/2022
"""

class StyleGAN(keras.Model):
    def __init__(self, epochs, batch_size):
        super(StyleGAN, self).__init__()
        self.discriminator = Discriminator().discriminator()
        self.generator = Generator().generator()
        self.epochs = epochs
        self.batch_size = batch_size

    def compile(self):
        """
        first time compile of the StyleGAN model
        """
        super(StyleGAN, self).compile()

        # set optimizers
        self.d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.g_optimizer=tf.keras.optimizers.Adam(learning_rate=1.25e-5)

        # set loss function
        self.loss_fn=tf.keras.losses.BinaryCrossentropy()

        # initialise logging
        self.discriminator_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss_metric = keras.metrics.Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.discriminator_loss_metric, self.generator_loss_metric]

    def plot_loss(self, history, relative_filepath):
        """
        plots saves and shows a loss graph of the StyleGAN training.
        param history: the model metrics logging
        param relative_filepath: the relative filepath for saving the plot ("" for no saving)
        """

        # pull metrics
        discriminator_loss_values = history.history['discriminator_loss']
        generator_loss_values = history.history['generator_loss']

        # plot values
        min_axis_value = min(min(generator_loss_values, discriminator_loss_values)) - 0.1
        max_axis_value = max(max(generator_loss_values, discriminator_loss_values)) + 0.1

        plt.plot(discriminator_loss_values, label='discriminator loss')
        plt.plot(generator_loss_values, label = 'generator loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([min_axis_value, max_axis_value])
        plt.legend(loc='upper right')

        # save and show plot
        if relative_filepath != "":
            dirname = os.path.dirname(__file__)
            filepath= os.path.join(dirname, relative_filepath)
            plt.savefig("{}\loss_plot.png".format(filepath))
            
        plt.show()
        

    def train(self, input_images_path="/keras_png_slices/", output_images_path="", images_count=3, weights_path="", plot_loss: bool=False):
        """
        trains the StyleGAN model over the specified epochs and builds the requested callbacks.
        param input_images_path: relative path for input image data
        param output_images_path: relative output image paths ("" for no saving)
        param images_count: number of images to save
        param weights_path: relative output weight path ("" for no saving)
        param plot_loss: True if loss should be plotted/saved, otherwise False
        """
        callbacks=[]

        # add image and model saving callbacks if paths are specified
        if output_images_path != "":
            callbacks.append(ImageSaver(output_images_path, images_count))
        if weights_path != "":
            callbacks.append(WeightSaver(weights_path))

        images = load_data(input_images_path)
        self.compile()
        history = self.fit(images, epochs=self.epochs, callbacks=callbacks)

        if plot_loss:
            self.plot_loss(history, output_images_path)
        

    @tf.function
    def train_step(self, real_images):
        """
        processes one epoch of StyleGAN training.
        param real_images: real image data
        output: dictionary of loss metrics
        """
        d_loss = self.train_discriminator(real_images)
        g_loss = self.train_generator()

        self.discriminator_loss_metric.update_state(d_loss)
        self.generator_loss_metric.update_state(g_loss)

        return {
            "discriminator_loss": self.discriminator_loss_metric.result(),
            "generator_loss": self.generator_loss_metric.result(),
        }

    def train_generator(self):
        """
        processes one epoch of generator training
        returns: generator loss
        """
        generator_inputs = self.get_generator_inputs()
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(generator_inputs)
            predictions = self.discriminator(fake_images)

            # the generator wants its fake images to be labelled as real (0)
            goal_labels = tf.zeros([self.batch_size, 1])
            g_loss = self.loss_fn(goal_labels, predictions)

            # apply gradients
            trainable_variables = self.generator.trainable_variables
            gradients = g_tape.gradient(g_loss, trainable_variables)
            self.g_optimizer.apply_gradients(zip(gradients, trainable_variables))

        return g_loss

    def train_discriminator(self, real_images):
        """
        processes one epoch of discriminator training
        real_images: real image data
        output: discriminator loss
        """
        generator_inputs = self.get_generator_inputs()
        generated_images = self.generator(generator_inputs)

        # Combine real and fake images, with corresponding labels fake -> 1, real -> 0
        images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, 1])], axis=0
        )

        # Train discriminator model
        with tf.GradientTape() as d_tape:
            predictions = self.discriminator(images)
            d_loss = self.loss_fn(labels, predictions)
            gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return d_loss

    def get_generator_inputs(self):
        """
        generates a set of inputs for the generator model, including latent z inputs, random noise inputs
        and a constant input layer
        returns: generator input list
        """
        # latent space noise for input into mapping
        z = [tf.random.normal((self.batch_size, 512)) for i in range(7)]

        # noise for B block inputs
        noise = [tf.random.uniform([self.batch_size, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]]

        # constant generator input
        input = tf.ones([self.batch_size, 4, 4, 512])

        return [input, z, noise]
