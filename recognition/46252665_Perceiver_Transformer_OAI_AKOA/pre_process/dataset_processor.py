"""

OAI AKOA Knee MRI Images
Osteoarthritis Initiative (OAI) Accelerated Osteoarthritis (OA) Knee MRI 2D
Images (~16K images, 1.7 GB). The caption of the images contained the
information regarding the left or right knee information. Original resolution
of the image were 260 x 288.


Processes OAI AKOA Dataset:
    * Creates Labels
    * Augments Data (switch-on function call for
                    processing dataset with augmentation)
    * Splits into Train, Test
    * Saves as .npy files

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import cv2
import imgaug.augmenters as iaa
import numpy as np
import os
import re
from settings.config import *
from sklearn.model_selection import train_test_split


class ProcessDataset:
    """Dataset labels generator and numpy file loader."""

    def __init__(self):
        """initialise variables"""
        self.load_images = []
        self.classes = []

        # training images
        self.x_train = None

        # test images
        self.x_test = None

        # training labels
        self.y_train = None

        # test labels
        self.y_test = None

        self.mixed_images = None

    def get_dataset(self):
        """walk over the image dataset directory, and search image
        caption with regex. If the caption contains
        left or l_e_f_t, it is assigned to 0. Else it is assigned 1.
        Finally load the image file, resize and append to a list."""
        for root, _, images in os.walk(PATH):
            for image in images:
                if image not in UNWANTED_FILES:
                    print(f'Image: {image}')
                    if re.search(fr'{RIGHT_TEXT}', image.lower()) or \
                            re.search(fr'{RIGHT_UNDERSCORE_TEXT}',
                                      image.lower()):
                        self.classes.append(RIGHT)
                    elif re.search(fr'{LEFT_TEXT}', image.lower()) or \
                            re.search(fr'{LEFT_UNDERSCORE_TEXT}',
                                      image.lower()):
                        self.classes.append(LEFT)

                    self.load_images.append(cv2.resize(
                        cv2.imread(PATH + image), IMAGE_SIZE))

    def augment_dataset(self):
        """Add image augmentation"""
        data_augmentation = iaa.Sequential([
            iaa.Fliplr(AUG_VAL), iaa.Flipud(AUG_FACTOR),
            iaa.GaussianBlur((0, BLUR_AUG))
        ])

        augmented_images = data_augmentation(
            images=np.array(self.load_images))
        self.mixed_images = self.load_images + \
                            [image for image in augmented_images]

    def save_dataset(self):
        """save dataset x_train, y_train, x_test and y_test as numpy file."""
        print('Saving Dataset!')
        np.save(f'{DATASET_PATH}/{X_TRAIN}{SAVE_EXT}', np.array(self.x_train))
        np.save(f'{DATASET_PATH}/{X_TEST}{SAVE_EXT}', np.array(self.x_test))
        np.save(f'{DATASET_PATH}/{Y_TRAIN}{SAVE_EXT}', np.array(self.y_train))
        np.save(f'{DATASET_PATH}/{Y_TEST}{SAVE_EXT}', np.array(self.y_test))

    def split_dataset(self):
        """Split dataset into training and test set."""
        self.x_train, self.x_test, \
        self.y_train, self.y_test = \
            train_test_split(
                self.mixed_images if self.mixed_images else self.load_images,
                self.classes + self.classes if self.mixed_images
                else self.classes,
                shuffle=True,
                test_size=TEST_SPLIT,
                random_state=RANDOM_STATE)

    def do_action(self):
        """sequential set of actions."""
        self.get_dataset()

        # switch-on this function call for processing dataset
        # with augmentation
        # self.augment_dataset()

        self.split_dataset()
        self.save_dataset()
