import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os

#Set up directories for dataset
test_directory = sorted(glob.glob( path + "\keras_png_slices_test\*.png"))
train_directory = sorted(glob.glob( path + "\keras_png_slices_train\*.png"))

train = tf.data.Dataset.from_tensor_slices(train_directory)
test = tf.data.Dataset.from_tensor_slices(test_directory)

train = train.shuffle(len(train_directory))
test = test.shuffle(len(test_directory))

# Pre processing and image displaying functions
# Map function

def pre_processing(input):
    # Read/output contents of filename
    data = tf.io.read_file(input)
    # Convert images to grayscale
    data = tf.image.decode_png(data, channels = 1)
    # Resize to 256*256
    data = tf.image.resize(data, (256,256))
    # Normalise to (-1,1)
    data = tf.cast(data, tf.float32) - 127.5
    data = data / 127.5
    return data

# Function to map datasets and set batch sizes

def map_and_batch(input, map_func, batch_size):
    input = input.map(map_func).batch(batch_size)
    return input


# Functions to display images


# Process single

def show_image(input):
    # Configure output figure's dpi and size
    output = plt.figure(dpi = 300, figsize = (4,4))
    # Iterate over input to display
    input_range = range(len(input))
    for i in input_range:
        plt.subplot(4,4, i + 1)
        for j in range(16):
            plt.axis("off")
            plt.imshow(input[i], cmap = "gray")
    plt.show()

# Process multiple

def show_images(input):
    # Iterate over a batch of images
    for image in input:
        show_image([tf.squeeze(image)][0])