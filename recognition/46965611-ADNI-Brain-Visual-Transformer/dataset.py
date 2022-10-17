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
            self.directory + "/train", labels='inferred', label_mode='int',
            image_size=[self.image_size, self.image_size],
            shuffle=True, batch_size=self.batch_size
        )

        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.directory + "/test", labels='inferred', label_mode='int', 
            image_size=[self.image_size, self.image_size],
            shuffle=True, batch_size=self.batch_size
        )

        # Augment data
        normalize = tf.keras.layers.Normalization()
        flip = tf.keras.layers.RandomFlip('horizontal')
        rotate = tf.keras.layers.RandomRotation(0.02)
        zoom = tf.keras.layers.RandomZoom(0.1, 0.1)

        train_data = train_data.map(
            lambda x, y: (rotate(flip(zoom(normalize(x)))), y)
        )

        test_data = test_data.map(
            lambda x, y: (rotate(flip(zoom(normalize(x)))), y)
        )

        # Take half of the 9000 images from the test set as validation data
        validation_data = test_data.take(9000//self.batch_size)

        # Use remaining 4500 images as test set
        test_data = test_data.skip(9000//self.batch_size)

        return train_data, validation_data, test_data