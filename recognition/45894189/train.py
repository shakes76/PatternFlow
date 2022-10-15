import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_data
from tensorflow import keras
from modules import Generator, Discriminator, WNetwork

class StyleGAN(keras.Model):
    def __init__(self):
        super(StyleGAN, self).__init__()
        self.discriminator = Discriminator().discriminator()
        self.generator = Generator().generator()
        self.mapping = WNetwork()

    def compile(self):
        super(StyleGAN, self).compile()
        self.d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_fn=tf.keras.losses.BinaryCrossentropy()
        self.discriminator_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss_metric = keras.metrics.Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.discriminator_loss_metric, self.generator_loss_metric]
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        g_loss = self.train_generator(batch_size)
        d_loss = self.train_discriminator(real_images, batch_size)

        self.generator_loss_metric.update_state(g_loss)
        self.discriminator_loss_metric.update_state(d_loss)

        return {
            "discriminator_loss": self.discriminator_loss_metric.result(),
            "generator_loss": self.generator_loss_metric.result(),
        }

    def train_generator(self, batch_size):
        z = tf.random.normal((batch_size, 256)) #TODO: Pass in latent_dim and stop magic numbers
        noise = [tf.random.normal([batch_size, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]] #TODO: Global variable these upscaling values somewhere
        input = tf.ones([32, 4, 4, 256])
        with tf.GradientTape() as g_tape:
            w = self.mapping(z)
            fake_images = self.generator([input, w, noise])
            predictions = self.discriminator(fake_images)

            goal_labels = tf.zeros([batch_size, 1])
            g_loss = self.loss_fn(goal_labels, predictions)

            trainable_weights = (self.mapping.trainable_weights + self.generator.trainable_weights)
        gradients = g_tape.gradient(g_loss, trainable_weights)
        self.g_optimizer.apply_gradients(zip(gradients, trainable_weights))

        return g_loss

    def train_discriminator(self, real_images, batch_size):
        z = tf.random.normal([batch_size, 256]) #TODO: Pass in latent_dim and stop magic numbers
        noise = [tf.random.normal([batch_size, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]] #TODO: Global variable these upscaling values somewhere
        input = tf.ones([32, 4, 4, 256])
        w = self.mapping(z)
        generated_images = self.generator([input, w, noise])

        # Combine real and fake, add labels with random noise
        images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train discriminator model
        with tf.GradientTape() as d_tape:
            predictions = self.discriminator(images)
            d_loss = self.loss_fn(labels, predictions)
        gradients = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        return d_loss



