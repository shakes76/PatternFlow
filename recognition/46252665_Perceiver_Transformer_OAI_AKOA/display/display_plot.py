"""
Plots random image and extracted patches from processed dataset

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from perceiver.extract_patches import Patches
from perceiver.load_data import LoadDataset
from perceiver.patch_encoder import PatchEncoder
from settings.config import *


class PlotPatches:
    """Visualise Patches"""
    def __init__(self):
        self.data = LoadDataset()

        self.image = self.data.x_test[np.random.choice(
            range(self.data.x_test.shape[0]))]

    def plot_sample(self):
        """Plot sample image"""
        plt.figure(figsize=(4, 4))
        plt.imshow(self.image.astype(np.uint8))
        plt.axis("off")
        plt.savefig(f'{FIGURE_LOCATION}sample.png')

    def plot_patches(self):
        """Plot patches of an sample image"""
        image_tensor = tf.convert_to_tensor([self.image])
        patches = Patches().call(image_tensor)

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")
        plt.savefig(f'{FIGURE_LOCATION}patches.png')

    def do_action(self):
        """sequential set of actions"""
        self.plot_sample()
        self.plot_patches()
