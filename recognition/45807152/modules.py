# Model of Improved UNet
import tensorflow as tf
from tensorflow.keras import Model


class ImprovedUNet():
    """
    An improved UNet UNet model based on the research paper.

    Ability to build CNN model.
    """

    def __init__(self, image_height, image_width, image_channels, filters,
                 kernel_size, alpha=0.01, padding='same', dropout=0.3,
                 upsample=(2, 2)):
        """Initialise new Improved UNet model."""
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.padding = padding
        self.dropout = dropout
        self.upsample = upsample
