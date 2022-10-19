import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import load_data
from tensorflow import keras
from modules import Generator, Discriminator, WNetwork
from util import ImageSaver, WeightSaver

class StyleGAN(keras.Model):
    def __init__(self, epochs):
        super(StyleGAN, self).__init__()
        self.discriminator = Discriminator().discriminator()
        self.generator = Generator().generator()
        self.epochs = epochs

    def compile(self):
        super(StyleGAN, self).compile()
        self.d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.g_optimizer=tf.keras.optimizers.Adam(learning_rate=1.25e-5)
        self.loss_fn=tf.keras.losses.BinaryCrossentropy()
        self.discriminator_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss_metric = keras.metrics.Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.discriminator_loss_metric, self.generator_loss_metric]

    def plot_loss(self, history, relative_filepath):
        discriminator_loss_values = history.history['discriminator_loss']
        generator_loss_values = history.history['generator_loss']

        min_axis_value = min(min(generator_loss_values, discriminator_loss_values)) - 0.1
        max_axis_value = max(max(generator_loss_values, discriminator_loss_values)) + 0.1

        plt.plot(discriminator_loss_values, label='discriminator loss')
        plt.plot(generator_loss_values, label = 'generator loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([min_axis_value, max_axis_value])
        plt.legend(loc='upper right')

        if relative_filepath != "":
            dirname = os.path.dirname(__file__)
            filepath= os.path.join(dirname, relative_filepath)
            plt.savefig("{}\loss_plot.png".format(filepath))
            
        plt.show()
        

    def train(self, input_images_path="/keras_png_slices/", output_images_path="", images_count=3, weights_path="", plot_loss: bool=False):
        callbacks=[]

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
        batch_size = tf.shape(real_images)[0]
        d_loss = self.train_discriminator(real_images, batch_size)
        g_loss = self.train_generator(batch_size)

        self.discriminator_loss_metric.update_state(d_loss)
        self.generator_loss_metric.update_state(g_loss)

        return {
            "discriminator_loss": self.discriminator_loss_metric.result(),
            "generator_loss": self.generator_loss_metric.result(),
        }

    def train_generator(self, batch_size):
        z = [tf.random.normal((batch_size, 512)) for i in range(7)]
        noise = [tf.random.uniform([batch_size, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]] #TODO: Global variable these upscaling values somewhere
        input = tf.ones([32, 4, 4, 512])
        with tf.GradientTape() as g_tape:
            fake_images = self.generator([input, z, noise])
            predictions = self.discriminator(fake_images)

            goal_labels = tf.zeros([batch_size, 1])
            g_loss = self.loss_fn(goal_labels, predictions)

            trainable_variables = self.generator.trainable_variables
            gradients = g_tape.gradient(g_loss, trainable_variables)
            self.g_optimizer.apply_gradients(zip(gradients, trainable_variables))

        return g_loss

    def train_discriminator(self, real_images, batch_size):
        z = [tf.random.normal((batch_size, 512)) for i in range(7)]
        noise = [tf.random.uniform([batch_size, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]] #TODO: Global variable these upscaling values somewhere
        input = tf.ones([32, 4, 4, 512])
        generated_images = self.generator([input, z, noise])

        # Combine real and fake, add labels with random noise
        images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=0
        )

        # Train discriminator model
        with tf.GradientTape() as d_tape:
            predictions = self.discriminator(images)
            d_loss = self.loss_fn(labels, predictions)
            gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return d_loss