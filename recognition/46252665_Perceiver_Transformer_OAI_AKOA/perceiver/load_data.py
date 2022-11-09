"""
Load processed dataset

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import numpy as np

from settings.config import *


class LoadDataset:

    def __init__(self):
        """Load training image arrays, labels and test image arrays,
        labels respectively."""
        self.x_train, self.y_train = \
            np.load(f'{DATASET_PATH}/{X_TRAIN}{SAVE_EXT}'), \
            np.load(f'{DATASET_PATH}/{Y_TRAIN}{SAVE_EXT}')

        self.x_test, self.y_test = np.load(
            f'{DATASET_PATH}/{X_TEST}{SAVE_EXT}'), \
            np.load(f'{DATASET_PATH}/{Y_TEST}{SAVE_EXT}')
