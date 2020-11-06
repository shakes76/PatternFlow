from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tfk2 = tf.keras.layers



def build_discriminator(input_shape):
    """
    Builds DCGAN disciminator model which makes used of max pooling
    layers to eventually flatten and have a single output layer

    Args:
        input_shape: The input shape that we expect
        

    Returns:
        Returns the discriminator model
    """
    input_layer = tf.keras.Input(input_shape)
    t = tfk2.Conv2D(32, 3, padding="same",
                    activation=tf.nn.leaky_relu)(input_layer)
    t = tfk2.Conv2D(32, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Conv2D(128, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.Conv2D(128, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Conv2D(256, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.Conv2D(256, 3, padding="same", activation=tf.nn.leaky_relu)(t)
    t = tfk2.MaxPool2D()(t)

    t = tfk2.Flatten()(t)
    t = tfk2.Dense(256, activation=tf.nn.leaky_relu)(t)
    t = tfk2.Dense(256, activation=tf.nn.leaky_relu)(t)
    # sigmoid is used since we expect 0 for fake, 1 for real
    t = tfk2.Dense(1, activation='sigmoid')(t)

    model = tf.keras.Model(inputs=input_layer, outputs=t)
    model.summary()
    return model



def build_generator(input_shape):
    """
    Builds DCGAN generator model which makes used of up sampling
    to build an image from random noise

    Args:
        directory: The input shape of the noise
        

    Returns:
        Returns the generator model
    """
    input_layer = tf.keras.Input(input_shape)
    t = tfk2.Dense(16*16*256)(input_layer)
    t = tfk2.Reshape((16,16,256))(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(256, 3, padding="same", activation=tf.nn.leaky_relu)(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(64, 3, padding="same", activation=tf.nn.leaky_relu)(t)

    t = tfk2.UpSampling2D()(t)
    t = tfk2.Conv2D(3, 3, padding="same", activation=tf.nn.leaky_relu)(t)

    t = tfk2.UpSampling2D()(t)
    #since images are normalised between [0,1], sigmoid is used
    t = tfk2.Conv2D(3, 3, padding="same", activation = 'sigmoid')(t)

    model = tf.keras.Model(inputs=input_layer, outputs=t)
    model.summary()
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    """
    Custom discrimnator loss function, making use of cross entropy

    Args:
        real_output: A real image from the datset
        fake_output: The generator's generated image (not real)
        

    Returns:
        Returns the loss of the discriminator
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Custom generator loss function, making use of cross entropy

    Args:
        fake_output: The generator's generated image (not real)
        

    Returns:
        Returns the loss of the generator
    """
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(data, disc, gen, opt1, opt2):
    """
    Training step to train the model

    Args:
        data: the batch of real images for the discriminator
        disc: discriminator model that has been built
        gen: generator model that has been built
        opt1: generator optimiser
        opt2: discriminator optimiser
        

    Returns:
        Returns the generator images
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #images are loaded in 32 batch size
        seed = tf.random.normal([32, 100])
        
        gen_image = gen(seed, training=True)
        real_output = disc(data, training=True)
        fake_output = disc(gen_image, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(\
            gen_loss, gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(\
            disc_loss, disc.trainable_variables)

        opt1.apply_gradients(zip(
            gradients_of_generator, gen.trainable_variables))
        opt2.apply_gradients(zip(
            gradients_of_discriminator, 
            disc.trainable_variables))

        return gen_image
