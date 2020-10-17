import glob
import time

import tensorflow as tf
import os
from recognition.s4436194_oasis_dcgan.data_helper import Dataset
from recognition.s4436194_oasis_dcgan.models_helper import make_generator_model, make_discriminator_model
from datetime import datetime
from tqdm import tqdm


DATA_TRAIN_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_train"
DATA_TEST_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_test"
DATA_VALIDATE_DIR = "keras_png_slices_data/keras_png_slices_data/keras_png_slices_seg_validate"

CHECKPOINT_DIR = "/training_checkpoints"


IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256
N_EPOCH_SAMPLES = 16
NOISE_DIMENSION = 100

tf.random.set_seed(3710)


class DCGANModelFramework:

    def __init__(self):
        # Instantiate discriminator and generator objects
        self.discriminator = make_discriminator_model(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.generator = make_generator_model()

        # Instantiate loss function
        self.generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Instantiate
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def train_dcgan(self, batch_size, epochs, verbose_iter=100):
        """Method for training the dcgan on the OASIS MRI images"""

        # Prepare dataset object
        dataset = Dataset(glob.glob(f"{DATA_TRAIN_DIR}/*.png"), IMAGE_WIDTH, IMAGE_HEIGHT)

        # Set the seed for all saved images, so we consistently get the same images
        seed = tf.random.normal([N_EPOCH_SAMPLES, NOISE_DIMENSION])

        # Set up checkpoints
        checkpoint_prefix = os.path.join(CHECKPOINT_DIR, f"checkpoint-{datetime.now().strftime('%Y-%m-%d')}")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        # Check for existing checkpoint, restore if possible
        if os.path.exists(checkpoint_prefix):
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

        @tf.function
        def train_step(images):
            noise = tf.random.normal([batch_size, NOISE_DIMENSION])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = self.generator_loss(self.generator_loss, fake_output)
                disc_loss = self.discriminator_loss(self.discriminator_loss, real_output, fake_output)

            # Calculate gradients
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Apply gradients
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_loss.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Main epoch loop
        total_batches = int((dataset.n_files / batch_size) + 1)
        for e in range(epochs):
            start = time.time()

            # Main training loop
            for i, batch_images in enumerate(dataset.get_batches(batch_size)):
                train_step(batch_images)

                if i % verbose_iter == 0:
                    g_loss, d_loss = self._get_current_loss(batch_size, batch_images)
                    print(f"Epoch {e}/{epochs}, Batch {i}/{total_batches} completed, Loss {g_loss} / {d_loss}")

            # Save the model every epoch
            checkpoint.save(file_prefix=checkpoint_prefix)

            print(f"Time for epoch {e + 1} is {(time.time() - start) / 60} minutes")

        print("Debug")

    def _get_current_loss(self, batch_size, images):
        noise = tf.random.normal([batch_size, NOISE_DIMENSION])
        generated_images = self.generator(noise, training=True)

        real_output = self.discriminator(images, training=True)
        fake_output = self.discriminator(generated_images, training=True)

        gen_loss = self.generator_loss(self.generator_loss, fake_output).numpy()
        disc_loss = self.discriminator_loss(self.discriminator_loss, real_output, fake_output).numpy()

        return round(gen_loss, 2), round(disc_loss, 2)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.discriminator_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        return self.generator_loss(tf.ones_like(fake_output), fake_output)

    def test_dcgan(self):
        pass
