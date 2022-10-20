# Model of Improved UNet
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, UpSampling2D


class ImprovedUNet():
    """
    Improved UNet model for lesion segmentation.

    An improved UNet model based on the "Brain Tumor Segmentation and
    Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge"
    research paper.

    Provides the ability to generate a UNet model for lesion segmentation
    with specified input image dimensions.
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

    def new_conv_layer(self, input, filters, kernel_size, strides):
        """Convolution layer for model."""
        conv_layer = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=self.padding,
                            activation=LeakyReLU(alpha=self.alpha))(input)

        return conv_layer

    def new_context_module(self, input, filters):
        """Context module for model."""
        first_conv_layer = self.new_conv_layer(input, filters,
                                               self.kernel_size, 1)
        second_conv_layer = self.new_conv_layer(first_conv_layer, filters,
                                                self.kernel_size, 1)
        dropout = Dropout(self.dropout)(second_conv_layer)

        return dropout

    def new_upsample_module(self, input, filters):
        """Upsample module for model."""
        upsample_layer = UpSampling2D(self.upsample)(input)
        conv_layer = self.new_conv_layer(upsample_layer, filters,
                                         self.kernel_size, 1)
        return conv_layer

    def new_localisation_module(self, input, filters):
        """Localisation module for model."""
        first_conv_layer = self.new_conv_layer(input, filters,
                                               self.kernel_size, 1)
        second_conv_layer = self.new_conv_layer(first_conv_layer, filters,
                                                self.kernel_size, 1)
        return second_conv_layer

    def new_segmentation_layer(self, input, filters):
        """Segmentation layer for model."""
        conv_layer = self.new_conv_layer(input, filters, (1, 1), 1)
        
        return conv_layer
