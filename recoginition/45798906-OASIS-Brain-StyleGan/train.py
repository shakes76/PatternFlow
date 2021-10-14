"""
    train.py

    This file contains functions that enables GAN training.

    Requirements:
        - Tensorflow 2.0
        - tqdm

    Author: Keith Dao
    Date created: 14/10/2021
    Date last modified: 14/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf
from tqdm import tqdm

# Loss functions
def generator_loss(fakes: tf.Tensor) -> float:

    return tf.keras.losses.BinaryCrossEntropy(tf.ones_like(fakes), fakes)


def discriminator_loss(reals: tf.Tensor, fakes: tf.Tensor) -> float:

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return (
        cross_entropy(tf.ones_like(reals), reals)
        + cross_entropy(tf.zeros_like(fakes), fakes)
    ) / 2


# Training functions
@tf.function
def train_step(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    real_images: tf.data.Dataset,
    latent_dimension: int,
    batch_size: int,
    img_size: int,
) -> tuple[float, float]:

    # Generate noise for the generator
    latent_noise = tf.random.normal([batch_size, latent_dimension])
    noise_images = tf.random.normal([batch_size, img_size, img_size, 1])

    # Train the models
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate some fake images
        fake_images = generator(
            [latent_noise, noise_images, tf.ones([batch_size, 1])]
        )

        # Use the discriminator to guess whether the images are real or fake
        real_guesses = discriminator(real_images)
        fake_guesses = discriminator(fake_images)

        # Calculate the losses
        disc_loss = discriminator_loss(real_guesses, fake_guesses)
        gen_loss = generator_loss(fake_guesses)

        # Calculate the gradient of the losses
        gradient_of_disc = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        gradient_of_gen = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )

        disc_optimizer.apply_gradients(
            zip(gradient_of_disc, discriminator.trainable_variables)
        )
        gen_optimizer.apply_gradients(
            zip(gradient_of_gen, generator.trainable_variables)
        )

    return gen_loss, disc_loss


def train(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    real_images: tf.data.Dataset,
    latent_dimension: int,
    batch_size: int,
    img_size: int,
    total_epochs: int,
    epoch_offset: int = 0,  # Number of previous completed epochs
    save_weights: bool = False,
    weight_save_path: bool = None,
    weight_save_interval: int = 5,
    save_images: bool = False,
    image_save_path: str = None,
) -> dict[str, list[float]]:

    if save_images:
        tf.io.gfile.makedirs(image_save_path)

    if save_weights:
        tf.io.gfile.makedirs(weight_save_path)

    history = {"gen": [], "disc": []}
    for epoch in range(total_epochs):

        # Save the losses for each batch
        gen_losses = []
        disc_losses = []

        for images in tqdm(real_images):
            gen_loss, disc_loss = train_step(
                generator,
                discriminator,
                gen_optimizer,
                disc_optimizer,
                real_images,
                latent_dimension,
                batch_size,
                img_size,
            )
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        history["gen"].append(tf.reduce_mean(gen_losses))
        history["disc"].append(tf.reduce_mean(disc_losses))

        print(
            f"Epoch {epoch+1+epoch_offset}: Generator Loss = {mean_gen_loss:.4f}, "
            f"Discriminator Loss = {mean_disc_loss:.4f}"
        )

        # Save one of the fake images
        # Generate noise for the generator
        if save_images:
            latent_noise = tf.random.normal([1, latent_dimension])
            noise_images = tf.random.normal([1, img_size, img_size, 1])
            save_img = tf.keras.preprocessing.image.array_to_img(
                generator([latent_noise, noise_images, tf.ones([1, 1])])[0]
            )
            save_img.save(
                f"{image_save_path}epoch-{epoch + epoch_offset + 1}.png"
            )

        # Save the weights
        if save_weights and (epoch + 1) % weight_save_interval == 0:
            generator.save_weights(
                f"{weight_save_path}generator/{epoch + epoch_offset + 1}"
            )
            discriminator.save_weights(
                f"{weight_save_path}discriminator/{epoch + epoch_offset + 1}"
            )

    return history
