import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow.keras.utils import to_categorical

# Import configuration parameters
from config import *


class ISIC_Dataset():
    """
    Dataset containing 2017 ISIC images and lesion masks.

    Can be split into training, validation, and test subsets for the
    purpose of model training and testing.
    """

    def __init__(self, data_path, mask_path,
                 image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
        # Store all image and mask paths in lists
        self.data_files = sorted(glob.glob(data_path))
        self.mask_files = sorted(glob.glob(mask_path))
        self.total_data_len = len(self.data_files)
        self.image_height = image_height
        self.image_width = image_width

    def get_data_splits(self, train_split, val_split, test_split):
        data = []
        masks = []

        for data_file in self.data_files:
            data_image = self.retrieve_processed_image(data_file)
            # Append data to list of data.
            data.append(data_image)

        for mask_file in self.mask_files:
            mask_detail = self.retrieve_processed_mask(mask_file)
            # Append mask to list of masks.
            masks.append(mask_detail)

        # Convert lists to numpy arrays.
        data = np.array(data)
        masks = np.array(masks)
        # One-hot encode mask details with 2 clases
        # Class 1 -> no lesion
        # Class 2 -> lesion
        masks = to_categorical(masks, num_classes=2)

        # Get training, validation and test split number of items
        train_size = round(train_split * self.total_data_len)
        val_size = round(val_split * self.total_data_len)
        test_size = round(test_split * self.total_data_len)

        # Split image and mask arrays into specified groups.
        self.train_x = data[:train_size]
        self.val_x = data[train_size:(train_size + val_size)]
        self.test_x = data[(train_size + val_size):]

        self.train_y = data[:train_size]
        self.val_y = data[train_size:(train_size + val_size)]
        self.test_y = data[(train_size + val_size):]

        # Return splits
        return ((self.train_x, self.train_y), (self.val_x, self.val_y),
                (self.test_x, self.test_y))

    def retrieve_processed_image(self, filepath):
        data_image = tf.io.read_file(filepath)
        data_image = tf.io.decode_jpeg(data_image, channels=3)

        # Resize
        data_image = tf.image.resize_with_pad(data_image, self.image_height,
                                              self.image_width)
        # Normalize
        data_image = data_image / 255.0

        return data_image

    def retrieve_processed_mask(self, filepath):
        mask_detail = tf.io.read_file(filepath)
        mask_detail = tf.io.decode_png(mask_detail, channels=1)

        # Resize
        mask_detail = tf.image.resize_with_pad(mask_detail, self.image_height,
                                               self.image_width)
        # Normalize
        mask_detail = mask_detail / 255.0

        return mask_detail
