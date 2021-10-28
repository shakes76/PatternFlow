"""
    gan.py

    This file contains the functions used to generate and train the generator and discriminator for StyleGAN.

    Requirements:
        - TensorFlow 2.0
        - tqdm
        - Matplotlib
        - util.py

    Author: Keith Dao
    Date created: 13/10/2021
    Date last modified: 28/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    add,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    Reshape,
    Resizing,
    UpSampling2D,
)
from tqdm import tqdm
from util import generate_image_grid, save_figure

# Custom layers
class AdaIN(Layer):
    """Adaptive Instance Normalisation Layer."""

    def __init__(self, epsilon: float = 1e-3):

        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape: list[tf.TensorShape]) -> None:

        dim = input_shape[0][-1]
        if dim == None:
            raise ValueError(
                f"Excepted axis {-1} of the input tensor be defined, but got an input with shape {input_shape}."
            )

        super(AdaIN, self).build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Apply the normalisation formula: gamma * (x - mean) / stddev + beta."""

        x, beta, gamma = inputs

        input_shape = x.shape
        axes = list(range(1, len(input_shape) - 1))
        mean = tf.math.reduce_mean(x, axes, keepdims=True)
        stddev = tf.math.reduce_std(x, axes, keepdims=True) + self.epsilon
        normalised = (x - mean) / stddev

        return normalised * gamma + beta


# ==========================================================
# Layer blocks
def gen_block(
    input: tf.Tensor,
    style: tf.Tensor,
    noise: tf.Tensor,
    filters: int,
    kernel_size: int,
    upSample: bool = True,
) -> tf.Tensor:
    def compute_random_input() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        beta = Dense(filters)(style)
        beta = Reshape([1, 1, filters])(beta)
        gamma = Dense(filters)(style)
        gamma = Reshape([1, 1, filters])(gamma)
        n = Dense(filters)(noise)
        return beta, gamma, n

    # Begin the generator block
    beta, gamma, n = compute_random_input()
    if upSample:
        out = UpSampling2D(interpolation="bilinear")(input)
        out = Conv2D(filters, kernel_size=kernel_size, padding="same")(out)
    else:
        out = Activation("linear")(input)
    out = add([out, n])
    out = AdaIN()([out, beta, gamma])
    out = LeakyReLU(0.20)(out)
    beta, gamma, n = compute_random_input()
    out = Conv2D(filters, kernel_size=kernel_size, padding="same")(out)
    out = add([out, n])
    out = AdaIN()([out, beta, gamma])
    out = LeakyReLU(0.2)(out)

    return out


def disc_block(
    input: tf.Tensor,
    filters: int,
    kernel_size: int,
    image_size: int,
    downSample: bool = True,
) -> tf.Tensor:

    # Begin the discriminator block
    out = input
    out = Conv2D(filters, kernel_size=kernel_size, padding="same")(out)
    out = Conv2D(filters, kernel_size=kernel_size, padding="same")(out)
    if downSample:
        out = Resizing(image_size // 2, image_size // 2)(out)
    out = LeakyReLU(0.2)(out)

    return out


# ==========================================================
# Models
def get_generator(
    latent_dim: int,
    output_size: int,
    num_filters: int,
    kernel_size: int,
) -> tf.keras.Model:

    STARTING_SIZE = 4

    # Inputs for each block
    mapping_inputs, noise_inputs = [], []
    curr_size = STARTING_SIZE
    while curr_size <= output_size:
        mapping_inputs.append(Input(shape=[latent_dim]))
        noise_inputs.append(Input(shape=[curr_size, curr_size, 1]))
        curr_size *= 2

    # Mapping network
    input_mapping = Input(shape=[latent_dim])
    mapping = input_mapping
    mapping_layers = 8
    for _ in range(mapping_layers):
        mapping = Dense(num_filters)(mapping)
        mapping = LeakyReLU(0.2)(mapping)
    mapping_network = tf.keras.Model(inputs=[input_mapping], outputs=mapping)

    # Generator network
    # Starting block
    curr_size = STARTING_SIZE
    input = Input(shape=[1])
    x = Lambda(lambda x: x * 0 + 1)(input)  # Set the constant value to be 1
    x = Dense(curr_size * curr_size * num_filters)(x)
    x = Reshape([curr_size, curr_size, num_filters])(x)
    x = gen_block(
        x,
        mapping_network(mapping_inputs[0]),
        noise_inputs[0],
        num_filters,
        kernel_size,
        upSample=False,
    )

    # Add upsampling blocks till the output size is reached
    block_num = 1
    curr_filters = num_filters
    while curr_size < output_size:
        curr_filters //= 2
        x = gen_block(
            x,
            mapping_network(mapping_inputs[block_num]),
            noise_inputs[block_num],
            curr_filters,
            kernel_size,
        )
        block_num += 1
        curr_size *= 2

    # To greyscale
    x = Conv2D(
        1, kernel_size=kernel_size, padding="same", activation="sigmoid"
    )(x)

    generator = tf.keras.Model(
        inputs=[mapping_inputs, noise_inputs, input], outputs=x
    )

    return generator


def get_discriminator(
    image_size: int,
    num_filters: int,
    kernel_size: int,
) -> tf.keras.Model:

    # Discriminator network
    input = Input(shape=[image_size, image_size, 1])
    x = input
    curr_size = image_size
    while curr_size > 4:
        x = disc_block(
            x, num_filters // (curr_size // 4), kernel_size, curr_size
        )
        curr_size //= 2
    x = disc_block(
        x,
        num_filters // (curr_size // 4),
        kernel_size,
        curr_size,
        downSample=False,
    )
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation("sigmoid")(x)

    discriminator = tf.keras.Model(inputs=[input], outputs=x)

    return discriminator


# ==========================================================
# Inputs
def generate_generator_inputs(
    latent_dimension: int,
    batch_size: int,
    img_size: int,
) -> list[tf.Tensor]:

    curr_size = 4
    mapping_inputs, noise_inputs = [], []
    while curr_size <= img_size:
        mapping_inputs.append(tf.random.normal([batch_size, latent_dimension]))
        noise_inputs.append(
            tf.random.uniform([batch_size, curr_size, curr_size, 1])
        )
        curr_size *= 2

    return [mapping_inputs, noise_inputs, tf.ones([batch_size, 1])]


# ==========================================================
# Optimisers
def get_optimizer(**hyperparameters) -> tf.keras.optimizers.Optimizer:

    return tf.keras.optimizers.Adam(**hyperparameters)


# ==========================================================
# Loss functions
def generator_loss(fakes: tf.Tensor) -> float:

    return tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fakes), fakes)


def discriminator_loss(reals: tf.Tensor, fakes: tf.Tensor) -> float:

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return (
        cross_entropy(tf.ones_like(reals), reals)
        + cross_entropy(tf.zeros_like(fakes), fakes)
    ) / 2


# ==========================================================
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

    generator_inputs = generate_generator_inputs(
        latent_dimension, batch_size, img_size
    )

    # Train the models
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate some fake images
        fake_images = generator(generator_inputs)

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
    batches: int,
    img_size: int,
    total_epochs: int,
    model_name: str,
    epoch_offset: int = 0,  # Number of previous completed epochs
    save_weights: bool = False,
    weight_save_path: str = None,
    weight_save_interval: int = 5,
    save_images: bool = False,
    image_save_path: str = None,
    image_save_interval: int = 1,
) -> tuple[list[float], list[float]]:

    if save_images:
        tf.io.gfile.makedirs(f"{image_save_path}{model_name}/")

    if save_weights:
        tf.io.gfile.makedirs(f"{weight_save_path}{model_name}/")

    gen_loss_history = []
    disc_loss_history = []
    for epoch in range(total_epochs):

        print(f"Epoch {epoch+1+epoch_offset}:")
        # Save the losses for each batch
        gen_losses = []
        disc_losses = []

        for images in tqdm(real_images.take(batches)):
            gen_loss, disc_loss = train_step(
                generator,
                discriminator,
                gen_optimizer,
                disc_optimizer,
                images,
                latent_dimension,
                batch_size,
                img_size,
            )
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        gen_loss_history.append(tf.reduce_mean(gen_losses))
        disc_loss_history.append(tf.reduce_mean(disc_losses))

        print(
            f"Generator Loss = {gen_loss_history[-1]:.4f}, "
            f"Discriminator Loss = {disc_loss_history[-1]:.4f}"
        )

        # Save some of the generated images
        # Generate noise for the generator
        if save_images and (epoch + 1) % image_save_interval == 0:
            images = generate_samples(
                generator, latent_dimension, batch_size, img_size
            )
            img_grid = generate_image_grid(images)
            save_figure(
                img_grid,
                f"{image_save_path}{model_name}/epoch-{epoch + epoch_offset + 1}.png",
            )

        # Save the weights
        if save_weights and (epoch + 1) % weight_save_interval == 0:
            generator.save_weights(
                f"{weight_save_path}{model_name}/generator/{epoch + epoch_offset + 1}"
            )
            discriminator.save_weights(
                f"{weight_save_path}{model_name}/discriminator/{epoch + epoch_offset + 1}"
            )

    return gen_loss_history, disc_loss_history


# ==========================================================
# Samples
def generate_samples(
    generator: tf.keras.Model,
    latent_dimension: int,
    sample_size: int,
    img_size: int,
) -> tf.Tensor:

    return generator(
        generate_generator_inputs(latent_dimension, sample_size, img_size)
    )
