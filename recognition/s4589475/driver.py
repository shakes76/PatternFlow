"""
Driver script which prepares the OASIS MRI Brain Dataset for use in the VQ-VAE algorithm.
The dataset is first loaded, then pre-processed, and finally set up inside a tensorflow Dataset object.
A training Dataset (batched), and both a batched and unbatched test Dataset are returned.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# OASIS Dataset constants
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 1 # images are in grayscale

def create_image(data):
    """
    Outputs a processed 3D image with extra grayscale channel added (4D tensor).
    The decoded original .png has pixel values normalised to a value between 0-1 and resized to 256x256.
    """
    #4D tensor now
    image = tf.io.decode_png(data, channels=CHANNELS)
    #normalise the image - float values between 0 and 1
    image = image / 255
    #resize image to 256 x 256
    return tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], antialias = True)

def get_image_from_file(path):
    """
    Extracts the file contents to output an image.
    """
    raw_data = tf.io.read_file(path)
    image = create_image(raw_data)
    return image

def set_dataset_parameters(dataset, batch_size):
    """
    Configures the dataset so it is in batches, well-shuffled and prefetches the images.
    """
    dataset = dataset.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def create_train_test_dataset(path_to_training_folder, path_to_test_folder, batch_size):
        
    """
    Creates the training and testing tensorflow Datasets from the images in the given filepaths.
    """
    training_files = tf.data.Dataset.list_files(path_to_training_folder, shuffle=False)
    test_files = tf.data.Dataset.list_files(path_to_test_folder, shuffle=False)

    #Retreive the size of each dataset
    num_train_images = training_files.cardinality().numpy()
    num_test_images = test_files.cardinality().numpy()
        
    # Shuffle the files
    training_files = training_files.shuffle(num_train_images, reshuffle_each_iteration=True)
    test_files = test_files.shuffle(num_test_images, reshuffle_each_iteration=True)

    # Map the filenames to images inside each dataset
    x_train = training_files.map(get_image_from_file, num_parallel_calls=tf.data.AUTOTUNE)
    x_test = test_files.map(get_image_from_file, num_parallel_calls=tf.data.AUTOTUNE)

    # Finalise data set configurations
    x_train = set_dataset_parameters(x_train, batch_size)
    x_test2 = x_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE).batch(544)
    x_test = set_dataset_parameters(x_test, batch_size)   
    
    #Return the datasets
    return x_train, x_test, x_test2