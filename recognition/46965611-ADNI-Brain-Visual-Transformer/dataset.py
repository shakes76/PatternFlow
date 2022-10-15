"""
dataset.py

Data loader for loading and preprocessing data.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
import tensorflow as tf

class DataLoader():
    def __init__(self, directory, image_size=128, batch_size=32):
        self.directory = directory
        self.image_size = image_size
        self.batch_size = batch_size

    def load_data(self):
        """
        Loads the dataset that will be used into Tensorflow datasets.
        """
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.directory + "/train", labels='inferred',
            image_size=[self.image_size, self.image_size],
            shuffle=True, batch_size=self.batch_size
        )

        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.directory + "/test", labels='inferred',
            image_size=[self.image_size, self.image_size],
            shuffle=True, batch_size=self.batch_size
        )

        # Take half of the 9000 images from the test set as validation data
        validation_data = test_data.take(4500)

        # Use remaining 4500 images as test set
        test_data = test_data.skip(4500).take(4500)

        return train_data, validation_data, test_data