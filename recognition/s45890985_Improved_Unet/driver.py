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
    mask_path = tf.strings.regex_replace(img_path, '-2_Training_Input', '_Training_GroundTruth')
    mask_path = tf.strings.regex_replace(mask_path, '.jpg', '_segmentation.png')

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
    image = tf.image.resize(image, (256, 256))
    mask = tf.image.resize(mask, (256, 256))
    return image, mask


def main():
    #directories and parameters of dataset
    data_path = os.path.join("D:/UQ/2021 Sem 2/COMP3710/Report", "ISIC_2018\ISIC2018_Task1-2_Training_Input")
    data_size = len(os.listdir(data_path))
    training_size = int(np.ceil(data_size * 0.8))
    testing_size = data_size - training_size
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 10

    # initiate full dataset containing all file directories
    full_dataset = tf.data.Dataset.list_files(data_path + "/*.jpg")
    full_dataset = full_dataset.shuffle(buffer_size=1000)

    # split full dataset into training and testing dataset with a 80-20 split
    train_dataset = full_dataset.take(training_size)
    test_dataset = full_dataset.skip(testing_size)

    # map dataset to img and mask
    train_dataset = train_dataset.map(load_img, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(load_img, num_parallel_calls=AUTOTUNE)

    # print dataset shapes
    print(train_dataset)
    print(test_dataset)

    # shuffle, batch and augment training dataset
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    # shuffle and batch testing dataset
    test = test.repeat()
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)



if __name__ == "__main__":
    main()