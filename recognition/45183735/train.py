import dataset
import modules
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


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

    # steps for training
    @tf.function
    def train_steps(self, images):
        z = [tf.random.uniform([self.batch_size, self.g_model.latent_size]) for i in range(7)]
        n = [tf.random.uniform([self.batch_size, 4 * 2 ** i, 4 * 2 ** i, 1]) for i in range(7)]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.g_model.model(tuple(z + n), training=True)
            real_out = self.d_model.d_model(images, training=True)
            fake_out = self.d_model.d_model(generated_images, training=True)
            g_loss = self.get_g_loss(fake_out)
            d_loss = self.get_d_loss(real_out, fake_out)

            gradients_of_generator = gen_tape.gradient(g_loss, self.g_model.model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.d_model.d_model.trainable_variables)

            self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.g_model.model.trainable_variables))
            self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.d_model.d_model.trainable_variables))

        return g_loss, d_loss

    def train(self, epochs):
        for epoch in range(epochs):
            m = 0
            # store the loss of generator and discriminator
            loss_g = []
            loss_d = []

            for image_batch in self.dataset:
                g_loss, d_loss = self.train_steps(image_batch)
                loss_g.append(g_loss)
                loss_d.append(d_loss)
                # calculate the precentage of progress
                progress = (len(image_batch) * m / (len(self.dataset) * len(image_batch))) * 100
                print("epoch:", epoch, ", percentage:", progress, "%", "gen_loss:", g_loss.numpy(), "disc_loss:",
                      d_loss.numpy())
                m = m + 1

            # save the loss for each epoch
            self.save_loss(epoch + 1, loss_g, loss_d, len(self.dataset))
            # save the images for each epoch
            self.save_images(self.g_model.model, epoch + 1, self.seed)

            # Save the model every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.ckpt.save(file_prefix=self.ckpt_prefix)

        # save the final images and loss after the final epoch

        self.save_images(self.g_model.model, epochs, self.seed)

    def save_loss(self, epoch, gen_loss1, disc_loss1, batch):

        batch_index = np.linspace(0, batch - 1, batch)
        plt.plot(batch_index, gen_loss1, "r", batch_index, disc_loss1, "g")
        plt.title('Loss at epoch {:04d}'.format(epoch))
        label = ["gen_loss", "disc_total_loss"]
        plt.legend(label)
        plt.xlabel('Image_batch_index')
        plt.ylabel('Loss')
        plt.savefig("loss_at_epoch_{:04d}.png".format(epoch))
        plt.close()

    def save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)
        plt.figure(figsize=(4, 4), constrained_layout=True)

        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow((predictions[i].numpy()), cmap="gray")
            plt.axis("off")

        plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
        plt.close()


if __name__ == "__main__":
    latent_size = 512
    input_size = 256
    batch_size = 8
    g_mapping = modules.G_Mapping(latent_size)
    g_s = modules.G_Synthesis(latent_size, g_mapping, input_size)
    g_style = modules.G_style(latent_size, input_size, g_s)
    discriminator = modules.Discriminator(input_size)
    dataset = dataset.Dataset("./keras_png_slices_data", batch_size, input_size)
    train = Train(dataset.train_ds, g_style, discriminator, input_size, batch_size)

    train.train(100)
