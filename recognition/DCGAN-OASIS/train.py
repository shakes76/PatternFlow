"""
A program to train a generative adversarial network on the OASIS dataset. 
Handles importing the data, initialising the model and training steps and
saves weights and images at each epoch.

@author Theo Duval
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob
from PIL import Image

from generator import Generator
from discriminator import Discriminator

import time


# Define some global variables and useful functions
EPOCHS = 500
BATCH_SIZE = 256

# Get all of the unlabelled images as there is no need for the partitioned
# data for this task
TEST = glob.glob("./keras_png_slices_data/keras_png_slices_test/*.png")
TRAIN = glob.glob("./keras_png_slices_data/keras_png_slices_train/*.png")
VALIDATE = glob.glob("./keras_png_slices_data/keras_png_slices_validate/*.png")
IMAGE_NAMES = TEST + TRAIN + VALIDATE

# Keep track of the loss function to be used
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Clear any previous models
tf.keras.backend.clear_session()

def load_images(filenames):
    """
    Loads in images for the current batch of filenames. This input is a sublist
    of IMAGE_NAMES of size BATCH_SIZE, and returns a 4D array of images with
    shape (BATCH_SIZE, x_size, y_size, colour_channels). Note that for this
    task, there is only one colour channel because the OASIS dataset is 
    greyscale.
    """

    # Initialise the results
    total = []

    # Iterate through each filename
    for i in range(len(filenames)):
        # Load the images
        image = Image.open(filenames[i])

        # Resize the images
        image = image.resize((64, 64))
        
        # Cast to an array
        image = np.array(image)

        # Add it to the list of images
        total.append(image)

    # Cast the result list to an array itself
    total = np.array(total)

    # Normalise the data
    total = total / 255.0

    # Add an axis to make this array 4D
    total = total[:, :, :, np.newaxis]

    return total


def generate_samples(generator, epoch):
    """
    Creates nine sample images with a given generator and saves them in a file. 
    Named per epoch, as one is generated at the end of each epoch to check on 
    the progress of training.
    """

    # Intialise the figure
    fig = plt.figure(figsize=(15,15))

    # Generate the seeds for each image
    seed = tf.random.normal([9, 100])

    for i in range(1, 10):
        # Choose the i'th image
        plt.subplot(3, 3, i)

        # Generate the image
        image = generator(seed, training=True)

        # Display the image
        plt.imshow(image[i-1])
        plt.axis('off')

    # Save the image
    plt.savefig("./generated_images/Epoch-{}.png".format(epoch+1))


def get_discrim_loss(real_out, fake_out):
    """
    Defines the loss function for the discriminator model. Uses binary cross
    entropy as there are only two classes of images (real vs fake). The loss
    needs to take into account the results of the discriminator on both the
    real and fake images.
    """

    real = BCE(tf.ones_like(real_out), real_out)
    fake = BCE(tf.zeros_like(fake_out), fake_out)

    return(real + fake)


def get_gen_loss(output):
    """
    Defines the loss function for the generator model. Also uses binary cross
    entropy.
    """
    return(BCE(tf.ones_like(output), output))


# Initialise both models
generator = Generator()
discriminator = Discriminator()

d_loss = tf.keras.metrics.Mean(name="discrim_loss")
g_loss = tf.keras.metrics.Mean(name="gen_loss")


@tf.function
def train_step(real_images):
    """
    Defines a single step of training for a batch of images. Calculates the
    loss for both models, applies the optimisers appropriately and records
    the loss for a historical record.
    """
    noise = tf.random.normal([BATCH_SIZE, 100])
    
    # Keep two tapes for each model
    with tf.GradientTape() as gen_tape:
        with tf.GradientTape() as discrim_tape:
            generated_images = generator(noise, training=True)
            
            decisions_real = discriminator(real_images, training=True)
            decisions_fake = discriminator(generated_images, training=True)
            
            discrim_loss = get_discrim_loss(decisions_real, decisions_fake)
            gen_loss = get_gen_loss(decisions_fake)
            
    # Get the respective gradients from the respective tapes
    gen_gradients = gen_tape.gradient(gen_loss, 
                                      generator.trainable_variables)

    discrim_gradients = discrim_tape.gradient(
            discrim_loss, discriminator.trainable_variables)
    
    # Save loss metrics
    d_loss(discrim_loss)
    g_loss(gen_loss)
    
    # Apply the gradients
    generator.optimiser.apply_gradients(zip(gen_gradients, 
                                            generator.trainable_variables))

    discriminator.optimiser.apply_gradients(
            zip(discrim_gradients, discriminator.trainable_variables))
    
    return(discrim_loss, gen_loss)
    
# List of ordered pairs with (d_loss, g_loss)
loss = []

for epoch in range(EPOCHS): 

    # Clear loss for this epoch    
    d_loss.reset_states()
    g_loss.reset_states()
    
    # Iterate through the dataset in batches of size BATCH_SIZE
    start_time = time.time()
    for i in range(0, len(IMAGE_NAMES), BATCH_SIZE):
        # Load the images
        images = load_images(IMAGE_NAMES[i:i+BATCH_SIZE])

        # Train
        (d_loss_step, g_loss_step) = train_step(images)
    
    # Save some samples
    generate_samples(generator, epoch)
    
    # Keep track of the loss for each model
    loss.append((d_loss.result(), g_loss.result()))
        
    # Print update messages
    print("Epoch {} computed in {} seconds".format(
            epoch+1, time.time()-start_time))
    print("Discriminator loss: {}".format(d_loss.result()))
    print("Generator loss: {}".format(g_loss.result()))

    # Save the weights of both models
    generator.save_weights("./saved_weights/gen_epoch{}".format(epoch + 1))
    discriminator.save_weights("./saved_weights/dis_epoch{}".format(epoch + 1))
