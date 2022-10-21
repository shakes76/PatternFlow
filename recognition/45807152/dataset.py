import tensorflow as tf
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
        """Specify filepaths to load data from.
        
        Establish image height and width for resizing.
        """
        # Store all image and mask paths in lists
        self.data_files = sorted(glob.glob(data_path))
        self.mask_files = sorted(glob.glob(mask_path))
        self.total_data_len = len(self.data_files)
        self.image_height = image_height
        self.image_width = image_width

    def get_data_splits(self, train_split, val_split, test_split):
        """
        Preprocess images and masks and return splits.
        
        Specify breakdown of training, validation and testing split size,
        function creates stacked tensors of data and applies processing
        as a mapping.
        """
        # Convert path lists to tensors.
        data_tensor = tf.convert_to_tensor(self.data_files)
        masks_tensor = tf.convert_to_tensor(self.mask_files)
        # Map image processing function to list of images/masks
        data_tensor = tf.map_fn(self.retrieve_processed_image, data_tensor,
                                dtype=tf.float32)
        masks_tensor = tf.map_fn(self.retrieve_processed_mask,
                                 masks_tensor, dtype=tf.float32)
        
        # One-hot encode mask details with 2 clases
        # Class 1 -> no lesion
        # Class 2 -> lesion
        masks_tensor = to_categorical(masks_tensor, num_classes=2)

        # Get training, validation and test split number of items
        train_size = round(train_split * self.total_data_len)
        val_size = round(val_split * self.total_data_len)
        test_size = round(test_split * self.total_data_len)

        # Split image and mask arrays into specified groups.
        self.train_x = data_tensor[:train_size]
        self.val_x = data_tensor[train_size:(train_size + val_size)]
        self.test_x = data_tensor[(train_size + val_size):]

        self.train_y = masks_tensor[:train_size]
        self.val_y = masks_tensor[train_size:(train_size + val_size)]
        self.test_y = masks_tensor[(train_size + val_size):]

        # Return splits
        return ((self.train_x, self.train_y), (self.val_x, self.val_y),
                (self.test_x, self.test_y))

    def retrieve_processed_image(self, filepath):
        """Retrieve pre-processed image from filepath."""
        data_image = tf.io.read_file(filepath)
        data_image = tf.io.decode_jpeg(data_image, channels=3)

        # Resize
        data_image = tf.image.resize_with_pad(data_image, self.image_height,
                                              self.image_width)
        # Normalize
        data_image = data_image / 255.0

        return data_image

    def retrieve_processed_mask(self, filepath):
        """Retrieve pre-processed mask details from filepath."""
        mask_detail = tf.io.read_file(filepath)
        mask_detail = tf.io.decode_png(mask_detail, channels=1)

        # Resize
        mask_detail = tf.image.resize_with_pad(mask_detail, self.image_height,
                                               self.image_width)
        # Normalize
        mask_detail = mask_detail / 255.0

        return mask_detail
