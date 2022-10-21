'''
Auxiliary Utilities for GAN.

This file contains a number of utility functions for loading and visualising data,
as well as calculating the SSIM between generated images and the testing set.

Requirements:
- TensorFlow
- Pillow
- glob
- numpy
- matplotlib

Author: Erik Brand
Date: 01/11/2020
License: Open Source
'''

import tensorflow as tf
from PIL import Image
import glob
import numpy as np
from matplotlib import pyplot

"""
Load a training dataset.
Returns a TensorFlow Dataset Iterator
"""
def load_data(filepath, batch_size):
    # Obtain image filepaths
    image_files = glob.glob(filepath + '*')
    # Load images
    images = np.array([np.array(Image.open(i).convert('L').resize((128,128))) for i in image_files])

    # Create 'channels' dimension
    images = images[:,:,:,np.newaxis]
    discriminator_input_dim = images.shape[1:]
    dataset_size = images.shape[0]

    # Normalise images 0-1
    images = images/255

    # Output trainign data shape for debugging
    print("Data Shape:")
    print(images.shape)

    # Create TensorFlow Dataset for efficient retreival of images
    # This is randomly shuffled and batched
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.shuffle(buffer_size=batch_size)
    images = images.repeat().batch(batch_size)
    image_iter = iter(images)

    return image_iter, discriminator_input_dim, dataset_size

"""
Load a testing dataset.
Returns a TensorFlow Dataset Iterator
"""
def load_test_data(filepath):
    # Obtain image filepaths
    image_files = glob.glob(filepath + '*')
    # Load images
    images = np.array([np.array(Image.open(i).convert('L').resize((128,128))) for i in image_files])

    # Create 'channels' dimension
    images = images[:,:,:,np.newaxis]
    dataset_size = images.shape[0]

    # Output test data shape for debugging
    print("Test Data Shape:")
    print(images.shape)

    # Create TensorFlow Dataset for efficient retreival of images
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.repeat()
    image_iter = iter(images)

    return image_iter, dataset_size

"""
Plot the loss history after training session
"""
def plot_history(disc_hist, gen_hist, output_path):
    # plot history
    pyplot.plot(disc_hist, label='loss_disc')
    pyplot.plot(gen_hist, label='loss_gen')
    pyplot.legend()
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Loss")
    pyplot.savefig(output_path + 'loss_plot.png')
    pyplot.close()

"""
Plot some example generated images in a grid
"""
def plot_examples(example_images, output_path):
    # Display example image shape for debugging
    print("Example Dim: ")
    print(example_images.shape)

    # Plot 4 x 4 grid of generated images
    dim = 4
    f, axarr = pyplot.subplots(dim,dim)
    for i in range(dim):
        for j in range(dim):
            axarr[i,j].imshow(example_images[dim * i + j], cmap='gray', vmin=0, vmax=255)
            axarr[i,j].axis('off')
    pyplot.savefig(output_path + 'example_output.png')
    pyplot.close()

"""
Calculate the SSIM over the testing set and the generated images
"""
def calculate_ssim(test_filepath, example_images, output_path):
    # Load the test dataset
    test_dataset, dataset_size = load_test_data(test_filepath)

    # Compute SSIM for each test image with each generated image
    ssims = []
    for i in range(example_images.shape[0]):
        for j in range(dataset_size):
            ssims.append(tf.image.ssim(test_dataset.get_next(), example_images[i], max_val=255))

    # Calculate average SSIM
    ssims = np.asarray(ssims)
    ssim = np.mean(ssims)

    # Save SSIM
    f = open(output_path + "SSIM.txt", 'w')
    f.write("SSIM: " + str(ssim))
    f.close()

    return ssim

"""
Generate example images using the trained generator
"""
def generate_example_images(gen, num_examples, latent_dim):
    # Generate some data from the latent space
    latent_data = tf.random.normal(shape=(num_examples, latent_dim))
    # Generate fake images
    fake_images = gen(latent_data)
    # Normalise fake images
    mins = tf.math.reduce_min(fake_images, axis=(1,2,3))[:,None,None,None]
    maxs = tf.math.reduce_max(fake_images, axis=(1,2,3))[:,None,None,None]
    fake_images = (fake_images - mins)/(maxs-mins)
    # Prepare the fake images for visualisation
    fake_images = fake_images*255
    fake_images = tf.cast(fake_images, dtype=tf.uint8)
    return fake_images