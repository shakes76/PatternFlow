import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow.keras.backend as Keras
import os
import cv2

class DataGenerator(Sequence):
    """Data Generator class for Keras. Inspiration taken from a Stanford tutorial and adapated to a classification problem.
       In order to improve model training time it's ideal to only load a portion of the dataset at once usign the generator."""

    def __init__(self, ids, labels, batch_size=32, image_size=128):
        """Generator initialization"""
        self.batch_size = batch_size # Training batch size, increase or decrease depending on memory needs
        self.ids = ids
        self.image_size = image_size # Defaults at 128x128, change if memory or time is an issue
        self.on_epoch_end()

    def __len__(self):
        "Number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Generate indexes of the batch
        if (idx + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - idx * self.batch_size

        batch = self.ids[idx*self.batch_size:(idx+1)*self.batch_size]

        img = []
        mask = []

        # For each directory name, load and preprocess the images and the masks using __load_image
        for image in batch:
            temp_img, tmp_mask = self.__load__(image)
            img.append(temp_img)
            mask.append(tmp_mask)

        return np.array(img), np.array(mask)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __load_image(self, id):
        """Loads the images from the dataset using the specified file path"""
        img_path = os.path.join(self.path, "images", id) + ".jpg" # training images
        mask_path = os.path.join(self.path, "masks", id) + "_segmentation.png" # training masks

        # Read in the image and resize to specified size
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = np.zeros((self.image_size, self.image_size, 1))
        # Read in the mask and resize it
        mask_ = cv2.imaread(mask_path, -1)
        mask_ = cv2.resize(mask_, (self.image_size, self.image_size))
        mask_ = np.expand_dims(mask_, axis=1)

        mask = np.maximum(mask, mask_)

        # Normalize to [0, 1] for activation function
        image = image/255.0
        mask = mask/255.0

        return image, mask

def dice_coefficient(y_true, y_predicted):
    y_true = np.asarray(y_true).astype(np.bool)
    y_predicted = np.asarray(y_predicted).astype(np.bool)
    intersect = np.logical_and(y_true, y_predicted)
    return (2.0 * intersect.sum()) / (y_true.sum() + y_predicted.sum())

def dice_coefficient_(y_true, y_predicted, smooth=1):
    y_true_flat = Keras.flatten(y_true)
    y_predicted_flat = Keras.flatten(y_predicted)
    intersect = Keras.sum(y_true_flat * y_predicted_flat)
    return (2.0 * intersect + smooth / (Keras.sum(y_true_flat) + Keras.sum(y_predicted_flat) + smooth))

def dice_loss(y_true, y_predicted):
    return -dice_coefficient(y_true, y_predicted)