import os
import re

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from settings.config import *


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

    def get_dataset(self):
        """walk over the image dataset directory, and search image caption with regex. If the caption contains
        left or l_e_f_t, it is assigned to 0. Else it is assigned 1.
        Finally load the image file, resize and append to a list."""
        for root, _, images in os.walk(PATH):
            for image in images:
                if image not in UNWANTED_FILES:
                    if re.search(fr'{RIGHT_TEXT}', image.lower()) or \
                            re.search(fr'{RIGHT_UNDERSCORE_TEXT}', image.lower()):
                        self.classes.append(RIGHT)
                    elif re.search(fr'{LEFT_TEXT}', image.lower()) or \
                            re.search(fr'{LEFT_UNDERSCORE_TEXT}', image.lower()):
                        self.classes.append(LEFT)

                    self.load_images.append(cv2.resize(cv2.imread(PATH + image), IMAGE_SIZE))

    def save_dataset(self):
        """save dataset x_train, y_train, x_test and y_test as numpy file."""
        print('Saving Dataset!')
        np.save(f'{DATASET_PATH}/{X_TRAIN}{SAVE_EXT}', np.array(self.x_train))
        np.save(f'{DATASET_PATH}/{X_TEST}{SAVE_EXT}', np.array(self.x_test))
        np.save(f'{DATASET_PATH}/{Y_TRAIN}{SAVE_EXT}', np.array(self.y_train))
        np.save(f'{DATASET_PATH}/{Y_TEST}{SAVE_EXT}', np.array(self.y_test))

    def split_dataset(self):
        """Split dataset into training and test set."""
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.load_images, self.classes,
                                                                                shuffle=True,
                                                                                test_size=TEST_SPLIT,
                                                                                random_state=RANDOM_STATE)

    def do_action(self):
        """sequential set of actions."""
        self.get_dataset()
        self.split_dataset()
        self.save_dataset()
