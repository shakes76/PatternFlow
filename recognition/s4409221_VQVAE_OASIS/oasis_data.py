import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img


class oasis_data(keras.utils.Sequence):
    """
    A class which iterates over the OASIS data and extracts the image information.
    
    """

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        """
        The constructor for extracting the image information.

        Parameters:
           batch_size (int): the batch size of the image.
           img_size (int) : the image size of the images.
           input_img_paths (list) : A list of file paths to the training dataset
           target_img_paths (list) : A list of file paths to the testing dataset.
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths      

    def __len__(self):
        """
        A function which returns the length of the object.

        Returns:
        Returns (int) length of the list object.

        """
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        A function which extracts the image information for the specified index.

        Parameters:
        idx (int) : an index of that describes which image is extracted from the list.

        Returns:
        x (Numpy array) : returns image information (pixel) of specified index (idx).

        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")

        #
        for j, path in enumerate(batch_input_img_paths):
          img = np.array(load_img(path, target_size=self.img_size))
          x[j] = img/255
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        #
        for j, path in enumerate(batch_target_img_paths):
          img = np.array(load_img(path, target_size=self.img_size, color_mode="grayscale"))
          one_hot = img == [0, 85, 170, 255]
          y[j] = one_hot
            
        # return x
        return x

