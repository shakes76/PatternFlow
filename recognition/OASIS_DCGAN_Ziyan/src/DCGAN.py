import os

import tensorflow as tf
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
from IPython import display


class DCGAN:

    def __init__(self, img_size, target_slice=None, input_shape=256, batch_size=256, noise_dim=256):
        """
        A DCGAN object that can create DCGAN structure and train. The object is flexible to different output size
        :param img_size: output size
        :param target_slice: Same as the parameter in imageLoader, the purpose here is when the slice is exist, the
            SSIM is using the mean value of between 256 comparisons. Else (Whole slices) the SSIM is using the max value
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
        self.target_slice = target_slice
        # Model initialize
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        # Checkpoint initialize
        if target_slice:
            self.checkpoint_dir = './models/' + 'chosen_' + str(len(target_slice)) + '_' + str(
                img_size) + '_training_checkpoints'
        else:
            self.checkpoint_dir = './models/' + 'chosen_whole_' + str(img_size) + '_training_checkpoints'
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

        model = tf.keras.Sequential()
        model.add(layers.Dense(int(self.img_size / 8) * int(self.img_size / 8) * self.batch_size, use_bias=False,
                               input_shape=(self.input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Reshape((int(self.img_size / 8), int(self.img_size / 8), self.batch_size)))
        assert model.output_shape == (None, int(self.img_size / 8), int(self.img_size / 8), self.batch_size)

        model.add(layers.Conv2DTranspose(512, (6, 6), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.img_size / 8), int(self.img_size / 8), 512)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(256, (6, 6), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.img_size / 4), int(self.img_size / 4), 256)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(128, (6, 6), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, int(self.img_size / 2), int(self.img_size / 2), 128)
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
            layers.Conv2D(128, (6, 6), strides=(2, 2), padding='same',
                          input_shape=[self.img_size, self.img_size, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (6, 6), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, (6, 6), strides=(2, 2), padding='same'))
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
        plt.figure(figsize=(2, 2))
        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.show()

    def train(self, dataset, epochs, patience=3):
        """
        A function that handles main part of training.
        :param dataset: The training dataset
        :param epochs: The epoch of training
        :param patience: Number of epochs achieves 0.6 SSIM after which training will be stopped.
        :return: hist_ssim: the history of ssim
        """
        hist_ssim = []
        num_examples_to_generate = 4
        seed = tf.random.normal([num_examples_to_generate, self.noise_dim])
        achieve_count = 0
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            display.clear_output(wait=True)
            self.generate_and_save_images(epoch + 1, seed)

            ssim = self.cal_ssim(dataset)
            print('SSIM: ', str(ssim))
            hist_ssim.append(ssim)

            # To get a reasonably clear image, SSIM test start after at least 500 epoch
            if epoch >= 1000:
                if ssim > 0.6:
                    achieve_count += 1
                    print(achieve_count)
                else:
                    achieve_count = 0

            # Save the model every 100 epoch
            if (epoch + 1) % 200 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            if achieve_count == patience:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                break

        display.clear_output(wait=True)
        self.generate_and_save_images(epochs, seed)
        return hist_ssim

    def load_model(self):
        """
        Load model from latest check point
        :return:
        """
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def generate_images(self, num=1, training=False):
        """
        A function that plot and return a number of fake images by the generator.
        :param training: If True, the function will not plot the image
        :param num: The number of fake images needs to generate
        :return: A series of fake images
        """
        noise = tf.random.normal([num, self.noise_dim])
        generated_image = self.generator(noise, training=False)
        if not training:
            plt.imshow(generated_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
        return generated_image.numpy() * 127.5 + 127.5

    def cal_ssim(self, real):
        """
        A function calculate the ssim between 256 fake and real images
        :param real: The real image tensor or dataset.
        :return: The mean SSIM between batch size number of fake and real images, if is the whole dataset the maximum
        """
        fake = self.generate_images(self.batch_size, training=True)
        if self.target_slice:
            return tf.image.ssim(fake, next(iter(real)).numpy().astype('float32')[0:self.batch_size] * 127.5 + 127.5,
                                 255).numpy().mean()
        else:
            return tf.image.ssim(fake, next(iter(real)).numpy().astype('float32')[0:self.batch_size] * 127.5 + 127.5,
                                 255).numpy().max()
