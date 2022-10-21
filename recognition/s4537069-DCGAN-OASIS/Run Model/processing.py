import tensorflow as tf
import glob
import os

def pre_processing_data(input):

    #Set up directories for dataset
    test_directory = sorted(glob.glob( input + "\keras_png_slices_test\*.png"))
    train_directory = sorted(glob.glob( input + "\keras_png_slices_train\*.png"))

    train = tf.data.Dataset.from_tensor_slices(train_directory)
    train = train.shuffle(len(train_directory))
    train_data = map_and_batch(train, pre_processing_dir, 16)
    take = train_data.take(50)
    images = tf.convert_to_tensor(list(take.as_numpy_iterator()))

    test = tf.data.Dataset.from_tensor_slices(test_directory)
    test = test.shuffle(len(test_directory))
    test_data = map_and_batch(test, pre_processing_dir, 16)

    return test_data, train_data, images



# Pre processing and image displaying functions
# Map function

def pre_processing_dir(input):
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


