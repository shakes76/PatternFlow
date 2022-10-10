import tensorflow as tf
import matplotlib.pyplot as plt
import os


# normalise the data between -1 to 1
def normalise(data):
    data = tf.cast(data/255., tf.float32)
    return data


class Dataset:

    def __init__(self, ds_path, batch_size, image_size):
        # path for the dataset folder
        self.path = ds_path
        self.batch_size = batch_size
        # image must be squared e.g. image_size = 256 -> image is 256x256
        self.image_size = image_size
        # dataset for training
        self.train_ds = self.get_train_ds()

    # get the training dataset from the path
    def get_train_ds(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.path + '/keras_png_slices_train',
            label_mode=None,
            color_mode='grayscale',
            batch_size=self.batch_size,
            image_size=(self.image_size, self.image_size)
        )

        train_ds = train_ds.map(normalise).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()

        return train_ds

    # get the test dataset from the path
    def get_test_ds(self):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.path + '/keras_png_slices_test',
            label_mode=None,
            color_mode='grayscale',
            batch_size=self.batch_size,
            image_size=(self.image_size, self.image_size)
        )
        test_ds = test_ds.map(normalise).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        return test_ds

    # get the validation dataset from the path
    def get_val_ds(self):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.path + '/keras_png_slices_validate',
            label_mode=None,
            color_mode='grayscale',
            batch_size=self.batch_size,
            image_size=(self.image_size, self.image_size)
        )
        val_ds = val_ds.map(normalise).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()
        return val_ds

