import tensorflow as tf

import os

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory


dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"

def import_dataset(batch_size, initial_image_shape, target_width, downsample_factor):
    '''
    Import the dataset.
    :param batch_size: batchsize to load the tf dataset with
    :param image_dim: image dimensions to crop the images to (these will be square)
    :return: training dataset
    :return: validation dataset
    :return: test dataset (list of paths to test image files)
    '''
    data_path = keras.utils.get_file(origin=dataset_url, fname="ADNI", extract=True)
    data_path = data_path[:-4]
    train_path = os.path.join(data_path, "AD_NC/train")
    test_path = os.path.join(data_path, "AD_NC/test")

    

    train = image_dataset_from_directory(
        train_path,
        batch_size=batch_size,
        image_size=initial_image_shape,
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
        crop_to_aspect_ratio=True
    )

    validation = image_dataset_from_directory(
        train_path,
        batch_size=batch_size,
        image_size=initial_image_shape,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
        crop_to_aspect_ratio=True
    )
    # scale images to (0,1)
    train = scale_images(train)
    validation = scale_images(validation)

    #pad images to 256x256
    train = train.map(lambda image :tf.image.pad_to_bounding_box(image, 0, 0, target_width, target_width))
    validation = validation.map(lambda image :tf.image.pad_to_bounding_box(image, 0, 0, target_width, target_width))

    # change train and validation datasets to YUV format and produce (input, output) tuple of (downscaled, original) images
    train = train.map(lambda x: (input_downsample(x, target_width, downsample_factor), input_process(x)))
    train = train.prefetch(buffer_size=32)

    validation = validation.map(lambda x: (input_downsample(x, target_width, downsample_factor), input_process(x)))

    # only look in the AD directory for images (test)
    image_path = os.path.join(test_path, "AD")
    test = collect_test_images(image_path)
    
    return train, validation, test

def scale_images(image_set):
    '''
    Scale from (0, 255) to (0, 1)
    '''
    scale = lambda img : img / 255
    return image_set.map(scale)


def input_downsample(input_image, target_width, downsample_factor=4):
  '''
  Downsample the images by a factor of up_sample_factor to generate low quality
  input images for the CNN
  '''
  input_image = input_process(input_image)
  output_size = [target_width // downsample_factor, target_width // downsample_factor]
  return tf.image.resize(input_image, output_size, method='area')

def input_process(input_image):
  '''
  Convert the images into the YUV colour space to make processing simpler for
  the GPU. Returns the greyscale channel of the YUV.
  '''
  input_image = tf.image.rgb_to_yuv(input_image)
  #split image into 3 subtensors along axis 3
  y, u, v = tf.split(input_image, 3, axis=3)
  #only return the y channel of the yuv (the greyscale)
  return y

def collect_test_images(image_path):
    '''
    Returns all paths to test images
    '''
    return [os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith('.jpeg')]

