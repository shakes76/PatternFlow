import dataset
import modules
import tensorflow as tf
import os



class Train:
    def __init__(self, dataset, g_model, d_model, input_size, batch_size):
        self.dataset = dataset
        # generator model
        self.g_model = g_model
        # discriminator model
        self.d_model = d_model
        # optimizer for discriminator
        self.d_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=.5)
        # optimizer for generator
        self.g_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=.5)
        self.input_size = input_size
        self.sample_num = 4
        self.seed = self.get_seed()
        # save checkpoint
        self.ckpt = self.get_ckpt()
        # checkpoint directory
        self.ckpt_prefix = os.path.join("./checkpoint", "ckpt")
        self.batch_size = batch_size

    # discriminator loss
    def get_d_loss(self, real_out, fake_out):
        # function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_l = cross_entropy(tf.ones_like(real_out), real_out)
        fake_l = cross_entropy(tf.zeros_like(fake_out), fake_out)
        total_l = real_l + fake_l
        return total_l

    # generator loss
    def get_g_loss(self, fake_out):
        # function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_out), fake_out)

    # use the seed to visualise
    def get_seed(self):
        return [tf.random.uniform([self.sample_num, self.g_model.latent_size]) for i in range(7)] \
               + [tf.random.uniform([self.sample_num, 4 * 2 ** i, 4 * 2 ** i, 1]) for i in range(7)]

    def get_ckpt(self):
        return tf.train.Checkpoint(
            generator_optimizer=self.g_optimizer,
            discriminator_optimizer=self.d_optimizer,
            generator=self.g_model.model,
            discriminator=self.d_model.d_model,
        )

