import matplotlib.pyplot as plt
import glob
import os
import time
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

from recognition.s4436194_oasis_dcgan.data_helper import Dataset
from recognition.s4436194_oasis_dcgan.models_helper import (
    make_generator_model,
    make_discriminator_model,
    make_generator_model_basic
)

DATA_TRAIN_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"
DATA_TEST_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_test"
DATA_VALIDATE_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_validate"

CHECKPOINT_DIR = "./training_checkpoints"

IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256
N_EPOCH_SAMPLES = 16
NOISE_DIMENSION = 100

tf.random.set_seed(3710)


class DCGANModelFramework:

    def __init__(self):

        self.save_name = f"{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(f"output/{self.save_name}/", exist_ok=True)
        os.makedirs(f"training_checkpoints/{self.save_name}/", exist_ok=True)

        # Instantiate discriminator and generator objects
        self.discriminator = make_discriminator_model(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.generator = make_generator_model_basic(NOISE_DIMENSION)

        # Instantiate loss function
        self.generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Instantiate
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def train_dcgan(self, batch_size, epochs, verbose_iter=100):
        """
        Method for training the dcgan on the OASIS MRI images

        Args:
            batch_size:
            epochs:
            verbose_iter:

        Returns:

        """

        # Prepare dataset object
        dataset = Dataset(glob.glob(f"{DATA_TRAIN_DIR}/*.png"), IMAGE_WIDTH, IMAGE_HEIGHT)

        # Set up checkpoints
        checkpoint_prefix = os.path.join(CHECKPOINT_DIR, f"{self.save_name}/ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        # Set the seed for all saved images, so we consistently get the same images
        seed = tf.random.normal([N_EPOCH_SAMPLES, NOISE_DIMENSION])

        # Check for existing checkpoint, restore if possible
        if os.path.exists(checkpoint_prefix):
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

        @tf.function
        def train_step(images):
            """

            Args:
                images:

            Returns:

            """
            noise = tf.random.normal([batch_size, NOISE_DIMENSION])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = self._compute_generator_loss(fake_output)
                disc_loss = self._compute_discriminator_loss(real_output, fake_output)

            # Calculate gradients
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Apply gradients
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Main epoch loop
        total_batches = int((dataset.n_files / batch_size) + 1)
        self.generate_and_save_images(0, seed)

        for e in range(epochs):
            start = time.time()

            # Main training loop
            for i, batch_images in tqdm(enumerate(dataset.get_batches(batch_size)), total=total_batches):
                train_step(batch_images)

                if i % verbose_iter == 0:
                    g_loss, d_loss = self._get_current_loss(batch_size, batch_images)
                    print(f"\nEpoch {e}/{epochs}, Batch {i}/{total_batches} completed, Loss {g_loss} / {d_loss}\n")

            # Save the model every epoch
            self.generate_and_save_images(e + 1, seed)
            checkpoint.save(file_prefix=checkpoint_prefix)

            print(f"\nTime for epoch {e + 1} is {(time.time() - start) / 60} minutes\n")

        print("Debug")

    def _get_current_loss(self, batch_size, images):
        """

        Args:
            batch_size:
            images:

        Returns:

        """
        noise = tf.random.normal([batch_size, NOISE_DIMENSION])
        generated_images = self.generator(noise, training=True)

        real_output = self.discriminator(images, training=True)
        fake_output = self.discriminator(generated_images, training=True)

        gen_loss = self._compute_generator_loss(fake_output).numpy()
        disc_loss = self._compute_discriminator_loss(real_output, fake_output).numpy()

        return round(gen_loss, 2), round(disc_loss, 2)

    def _compute_discriminator_loss(self, real_output, fake_output):
        """

        Args:
            real_output:
            fake_output:

        Returns:

        """
        real_loss = self.discriminator_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _compute_generator_loss(self, fake_output):
        """
        Return the generator loss based on the fake output

        Args:
            fake_output:

        Returns:

        """
        return self.generator_loss(tf.ones_like(fake_output), fake_output)

    def generate_and_save_images(self, epoch, test_input):
        """
        Create a set of test images, designed to do this at each epoch

        Args:
            epoch: Number of epochs completed
            test_input: Array of noise to generate image from
        """

        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)

            image = predictions[i, :, :, 0].numpy()
            image = (((image - image.min()) * 255) / (image.max() - image.min()))
            plt.imshow(image, cmap="Greys")
            plt.axis('off')

        plt.savefig("output/{}/image_at_epoch_{:04d}.png".format(self.save_name, epoch))
        plt.close()

    def test_dcgan(self):
        # TODO implement this
        pass
