"""
OASIS DCGAN Implementation framework

Framework for training and testing a DCGAN model. Two core classes are specified in this file,
  * Dataset: handles importing data and dealing it out in batches to mitigate large memory requirements
  * DCGANModelFramework: handles training and testing (creating images) for an input generator and
      discriminator model

Note that this automatically creates saving folders for output images (at each epoch), and model checkpoints. This
naming convention depends on the date being run and the size of the model.

@author nthompson97

Original GAN paper: https://arxiv.org/pdf/1511.06434.pdf
"""

import glob

import itertools
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from typing import List

DATA_TRAIN_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_train"
DATA_TEST_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"
DATA_VALIDATE_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate"

CHECKPOINT_DIR = "./training_checkpoints"

N_EPOCH_SAMPLES = 16
NOISE_DIMENSION = 100  # {100, 256}

tf.random.set_seed(3710)


class Dataset:
    """
    Class used to read raw data and return in a batch format.
    """

    def __init__(self, data_files: List[str], image_width: int, image_height: int):
        self.data_files = data_files
        self.n_files = len(data_files)

        self.image_width = image_width
        self.image_height = image_height

    def get_batches(self, batch_size: int) -> tf.Tensor:
        """
        Serve data batches, yield so this can be called as a loop

        Args:
            batch_size: The first dimension of the output array representing the batch size

        Yields:
            (batch_size, self.image_width, self.image_height, 1) numpy array representing a batch
        """
        idx = 0
        while idx + batch_size <= self.n_files:
            batch = self._get_batch(self.data_files[idx: idx + batch_size])
            idx += batch_size
            yield -1 + 2 * (batch - tf.reduce_min(batch)) / (tf.reduce_max(batch) - tf.reduce_min(batch))

    def _get_batch(self, image_files: List[str]) -> tf.Tensor:
        """
        Return an array of images in the form of a batch

        Args:
            image_files: List of file paths of images to read

        Returns:
            (batch_size, self.image_width, self.image_height, 1) numpy array representing batch
        """

        images = [self._get_image(sample_file) for sample_file in image_files]
        batch = tf.cast(tf.stack(images), dtype=tf.float32)

        # Add channel dimension
        batch = tf.reshape(batch, batch.shape + (1,))

        return batch

    def _get_image(self, image_path: str) -> tf.Tensor:
        """
        Returns the numpy array of a single image. Also responsible for resizing the image appropriately
        using PIL image resize

        Args:
            image_path: Single path to image

        Returns:
            (self.image_width, self.image_height) numpy array representing an image

        """
        image = Image.open(image_path).resize((self.image_width, self.image_height))
        assert image.size == (self.image_width, self.image_height), f"Inconsistent image size: {image.size}"

        return tf.reshape(tf.constant(image.convert(mode="L").getdata()), (self.image_width, self.image_height))


class DCGANModelFramework:
    """
    Class used to train and test a GAN framework using tensorflow
    """

    def __init__(self, discriminator, generator, size):

        # Instantiate discriminator and generator objects
        self.discriminator = discriminator
        self.generator = generator
        self.size = size

        # Set the seed for all saved images, so we consistently get the same images
        self.seed = tf.random.normal([N_EPOCH_SAMPLES, NOISE_DIMENSION])

        # Set uo save name and required directories
        self.save_name = f"{datetime.now().strftime('%Y-%m-%d')}-{self.size}x{self.size}"
        os.makedirs(f"output/{self.save_name}/", exist_ok=True)
        os.makedirs(f"training_checkpoints/{self.save_name}/", exist_ok=True)

    def train_dcgan(self, batch_size: int, epochs: int):
        """
        Method for training the dcgan on the OASIS MRI images

        Args:
            batch_size: Number of images to compose a single training batch
            epochs: Number of epochs to train the data over
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
            The train step of the main model. Uses gradient tape to evaluate loss and update optimisers

            Args:
                images: (BATCH_SIZE, self.size, self,size, 1) tensor of images to train on
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
        self.generate_and_image_subplot(checkpoint_epoch)

        # Start from existing epochs
        for e in range(checkpoint_epoch, epochs + checkpoint_epoch):
            start = time.time()

            # Main training loop
            for i, batch_images in tqdm(enumerate(dataset.get_batches(batch_size)), total=total_batches):
                train_step(batch_images)

                if i % (total_batches / 5) == 0:
                    self.evaluate_ssim(dataset)

            # Save the model every epoch
            self.generate_and_image_subplot(e + 1)
            checkpoint.save(file_prefix=checkpoint_prefix)

            print(f"\nTime for epoch {e + 1} is {(time.time() - start) / 60} minutes\n")

    def _compute_discriminator_loss(self, real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
        """
        Calculate loss for the discriminator model. Loss for true images is compared to an array of ones,
        and loss for the fake image is against an array of zeros.

        Args:
            real_output: (BATCH_SIZE, 1) tensor of a real images classifications
            fake_output: (BATCH_SIZE, 1) tensor of a generated images classifications

        Returns:
            Loss as specified for the discriminator model
        """
        real_loss = self.discriminator.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _compute_generator_loss(self, fake_output: tf.Tensor) -> tf.Tensor:
        """
        Return the generator loss based on the fake output. Loss is compared to a tensor of ones

        Args:
            fake_output: (BATCH_SIZE, 1) tensor of a generated images classifications

        Returns:
            Loss as specified for the generator model
        """
        return self.generator.loss(tf.ones_like(fake_output), fake_output)

    def evaluate_ssim(self, dataset: Dataset):
        """
        Compute structure similarity for a batch of generated images against the true dataset. Iterate over different
        combinations of true and generated images to find a sounds statistic for ssim with a reduced variance

        https://github.com/w13b3/SSIM-py/blob/master/image/ssim.py

        Args:
            dataset: Dataset object used to get true images
        """

        generated = self.generator(tf.random.normal([16, NOISE_DIMENSION]))
        true = next(dataset.get_batches(16))

        ssim_track = [self._structural_similarity(generated[i, :, :, 0], true[i, :, :, 0])
                      for i, j in itertools.combinations(range(16), 2)]

        print(f"\nStructural Similarity: {sum(ssim_track) / len(ssim_track)}")

    @staticmethod
    def _structural_similarity(arr_1: tf.Tensor, arr_2: tf.Tensor) -> float:
        """
        Compute structural similarity between two tensors

        https://en.wikipedia.org/wiki/Structural_similarity

        Args:
            arr_1: (self.size, self,size) tensor
            arr_2: (self.size, self,size) tensor

        Returns:
            Structural similarity index between the two tensors
        """
        mu_1, mu_2 = arr_1, arr_2
        sigma_11 = tf.multiply(arr_1, arr_1)
        sigma_22 = tf.multiply(arr_2, arr_2)
        sigma_12 = tf.multiply(arr_1, arr_2)

        # Compute weighted variances
        mu_11 = tf.multiply(mu_1, mu_1)
        mu_22 = tf.multiply(mu_2, mu_2)
        mu_12 = tf.multiply(mu_1, mu_2)
        sigma_11 = tf.subtract(sigma_11, mu_11)
        sigma_22 = tf.subtract(sigma_22, mu_22)
        sigma_12 = tf.subtract(sigma_12, mu_12)

        # constants to avoid numerical instabilities close to zero
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2

        # SSIM (contrast sensitivity)
        ssim = ((2 * mu_12 + c1) * (2.0 * sigma_12 + c2)) / ((mu_11 + mu_22 + c1) * (sigma_11 + sigma_22 + c2))
        return tf.reduce_mean(ssim).numpy()

    def generate_and_image_subplot(self, epoch):
        """
        Create a set of test images, designed to do this at each epoch

        Args:
            epoch: Number of epochs completed
        """

        predictions = self.generator(self.seed, training=False)

        fig = plt.figure(figsize=(8, 8), dpi=150)

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

        Args:
            save_dir: The directory the tensorflow checkpoint is saved under
        """

        save_name = save_dir if save_dir is not None else self.save_name
        assert os.path.exists(os.path.join(CHECKPOINT_DIR, f"{save_name}")), f"Directory does not exist: {save_name}"

        # Load checkpoints
        checkpoint_prefix = os.path.join(CHECKPOINT_DIR, f"{save_name}/")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer,
                                         discriminator_optimizer=self.discriminator.optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

        # Set the seed for all saved images, so we consistently get the same images
        input_ = tf.random.normal([1, 100])
        output = self.generator(input_)

        # Rescale tensor to 0 - 255 integer range
        output = tf.math.round((output - tf.reduce_min(output)) / (tf.reduce_max(output) - tf.reduce_min(output)) * 255)

        # Plot the generated image
        plt.imshow(output.numpy()[0, :, :, 0], cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.show()
