import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.utils import Sequence


def load_img(img_path):
    # method taken and derived from Lab2 part 3 code
    # read and decode image to uint8 array
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # change img_path into coresponding mask_path
    mask_path = tf.strings.regex_replace(img_path, '_Data', '_GroundTruth')
    mask_path = tf.strings.regex_replace(mask_path, '.jpg', '_Segmentation.png')

    # read and decode mask to uint8 array
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask)

    # one hot encoding
    mask = tf.where(mask == 0, np.dtype('uint8').type(0), mask)
    mask = tf.where(mask != 0, np.dtype('uint8').type(1), mask)

    # normalise and reshape data
    # convert values from 0-255 to 0 - 1
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)

    # resize image to dims
    image = tf.image.resize(image, (512, 512))
    mask = tf.image.resize(mask, (512, 512))
    return image, mask

def aug_img(img, mask):
    pass
def main():



if __name__ == "__main__":
    main()