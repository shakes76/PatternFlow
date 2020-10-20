"""
Questions for tutorial
- What's the difference between the project and demo 2? wrt gans
- What do we need to consider wrt model design
- How do we calculate SSIM for a single image?
- How can I run jobs through the uq server to save time?

https://medium.com/deep-dimension/gans-a-modern-perspective-83ed64b42f5c
"""

import matplotlib.pyplot as plt
import glob
import os
import time
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm

from recognition.s4436194_oasis_dcgan.data_helper import Dataset
from recognition.s4436194_oasis_dcgan.models_helper import (
    make_models_28,
    make_models_64,
    make_models_128,
)

DATA_TRAIN_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_train"
DATA_TEST_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"
DATA_VALIDATE_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate"

CHECKPOINT_DIR = "./training_checkpoints"

N_EPOCH_SAMPLES = 16
NOISE_DIMENSION = 100

tf.random.set_seed(3710)


class DCGANModelFramework:

    def __init__(self):

        # Instantiate discriminator and generator objects
        self.discriminator, self.generator, self.size = make_models_128()

        # Set the seed for all saved images, so we consistently get the same images
        self.seed = tf.random.normal([N_EPOCH_SAMPLES, NOISE_DIMENSION])

        # Need to pass an example through first
        _ = self.discriminator(self.generator(self.seed))
        self.generator.summary()
        self.discriminator.summary()

        # Set uo save name and required directories
        self.save_name = f"{datetime.now().strftime('%Y-%m-%d')}-{self.size}x{self.size}"
        os.makedirs(f"output/{self.save_name}/", exist_ok=True)
        os.makedirs(f"training_checkpoints/{self.save_name}/", exist_ok=True)

    def train_dcgan(self, batch_size, epochs):
        """
        Method for training the dcgan on the OASIS MRI images

        Args:
            batch_size:
            epochs:

        Returns:

        """

        # Prepare dataset object
        dataset = Dataset(glob.glob(f"{DATA_TRAIN_DIR}/*.png"), self.size, self.size)

        # Set up checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, self.save_name)
        checkpoint_prefix = f"{checkpoint_path}/ckpt"
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer,
                                         discriminator_optimizer=self.discriminator.optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        # Check for existing checkpoint, restore if possible
        if glob.glob(f"{checkpoint_path}/*.index"):
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
            checkpoint_epoch = max(int(i[-7]) for i in glob.glob(f"{checkpoint_path}/*.index"))
            print(f"Reverted to checkpoint: {checkpoint_prefix}, epoch: {checkpoint_epoch}")
        else:
            checkpoint_epoch = 0

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
            self.generator.optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator.optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Main epoch loop
        total_batches = int((dataset.n_files / batch_size) + 1)
        self.generate_and_save_images(0)

        # Start from existing epochs
        for e in range(checkpoint_epoch, epochs + checkpoint_epoch):
            start = time.time()

            # Main training loop
            for i, batch_images in tqdm(enumerate(dataset.get_batches(batch_size)), total=total_batches):
                train_step(batch_images)

            # Save the model every epoch
            self.generate_and_save_images(e + 1)
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
        real_loss = self.discriminator.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _compute_generator_loss(self, fake_output):
        """
        Return the generator loss based on the fake output

        Args:
            fake_output:

        Returns:

        """
        return self.generator.loss(tf.ones_like(fake_output), fake_output)

    def generate_and_save_images(self, epoch):
        """
        Create a set of test images, designed to do this at each epoch

        Args:
            epoch: Number of epochs completed
        """

        predictions = self.generator(self.seed, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)

            image = predictions[i, :, :, 0].numpy()
            image = (((image - image.min()) * 255) / (image.max() - image.min()))
            plt.imshow(image, cmap="Greys")
            plt.axis('off')

        plt.savefig("output/{}/image_at_epoch_{:04d}.png".format(self.save_name, epoch))
        plt.close()

    def test_dcgan(self, save_dir=None):
        """
        Generate and show a generated image.

        A model checkpoint must be saved locally, either under the instantiated framework save name or
        through the supplied kwarg

        Returns:
        """

        save_name = save_dir if save_dir is not None else self.save_name
        assert os.path.exists(os.path.join(CHECKPOINT_DIR, f"{save_name}")), f"Directory does not exist: {save_name}"

        # Load checkpoints
        checkpoint_prefix = os.path.join(CHECKPOINT_DIR, f"{self.save_name}/")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer,
                                         discriminator_optimizer=self.discriminator.optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

        # Set the seed for all saved images, so we consistently get the same images
        input_ = tf.random.normal([1, 100])
        output = self.generator(input_)

        # Plot the generated image
        plt.imshow(output.numpy()[0, :, :, 0], cmap="Greys")
        plt.show()

        print("Done")

