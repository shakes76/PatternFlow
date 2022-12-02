# Using tensorflow version 2.9.2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, UpSampling2D, Input, Add, concatenate
from tensorflow.keras import Model

"""
Class that contains the improved unet model as described in the paper
"""
class ImprovedUnetModel:
    def __init__(self, rows = 128, columns = 128, channels = 1, filters = 16, kernel = (3, 3)):
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.filters = filters
        self.kernel = kernel
        self.model = None
        self.dropout = 0.3
        self.alpha = 0.01
        self.padding = "same"
        self.upsample = (2, 2)

    def new_conv(self, input_layer, filters, kernel, strides):
        """
        Creates a new convolutional layer
        """
        conv_layer = Conv2D(filters = filters, kernel_size = kernel, strides = strides,
                            padding = self.padding, activation = LeakyReLU(alpha = self.alpha))(input_layer)
        return conv_layer

    def context_module(self, input_layer, filters):
        """
        Creates a context module
        """
        conv_layer1 = self.new_conv(input_layer, filters, self.kernel, 1)
        conv_layer2 = self.new_conv(conv_layer1, filters, self.kernel, 1)
        dropout1 = Dropout(self.dropout)(conv_layer2)
        return dropout1

    def upsampling_module(self, input_layer, filters):
        """
        Creates an upsampling module
        """
        upsample_layer1 = UpSampling2D(self.upsample)(input_layer)
        conv_layer1 = self.new_conv(upsample_layer1, filters, self.kernel, 1)
        return conv_layer1

    def localization_module(self, input_layer, filters):
        """
        Creates a localization module
        """
        conv_layer1 = self.new_conv(input_layer, filters, self.kernel, 1)
        conv_layer2 = self.new_conv(conv_layer1, filters, (1, 1), 1)
        return conv_layer2

    def segmentation_layer(self, input_layer, filters):
        """
        Creates a segmentation layer
        """
        conv_layer1 = self.new_conv(input_layer, filters, (1, 1), 1)
        return conv_layer1

    def build_model(self):
        """
        Builds the unet model
        """
        # Going down the U
        # Input Layer
        input_layer = Input(shape = (self.rows, self.columns, self.channels))

        # Level 1
        conv1 = self.new_conv(input_layer, self.filters, self.kernel, 1)
        context1 = self.context_module(conv1, self.filters)
        add1 = Add()([conv1, context1])

        # Level 2
        conv2 = self.new_conv(add1, self.filters * 2, self.kernel, 2)
        context2 = self.context_module(conv2, self.filters * 2)
        add2 = Add()([conv2, context2])

        # Level 3
        conv3 = self.new_conv(add2, self.filters * 4, self.kernel, 2)
        context3 = self.context_module(conv3, self.filters * 4)
        add3 = Add()([conv3, context3])

        # Level 4
        conv4 = self.new_conv(add3, self.filters * 8, self.kernel, 2)
        context4 = self.context_module(conv4, self.filters * 8)
        add4 = Add()([conv4, context4])

        # Level 5
        conv5 = self.new_conv(add4, self.filters * 16, self.kernel, 2)
        context5 = self.context_module(conv5, self.filters * 16)
        add5 = Add()([conv5, context5])
        upsample1 = self.upsampling_module(add5, self.filters * 8)

        # Going up the U
        # Level 4
        concat1 = concatenate([upsample1, add4])
        localization1 = self.localization_module(concat1, self.filters * 8)
        upsample2 = self.upsampling_module(localization1, self.filters * 4)

        # Level 3
        concat2 = concatenate([upsample2, add3])
        localization2 = self.localization_module(concat2, self.filters * 4)
        segmentation1 = self.segmentation_layer(localization2, self.filters)
        segmentation1up = UpSampling2D()(segmentation1)
        upsample3 = self.upsampling_module(localization2, self.filters * 2)

        # Level 2
        concat3 = concatenate([upsample3, add2])
        localization3 = self.localization_module(concat3, self.filters * 2)
        segmentation2 = self.segmentation_layer(localization3, self.filters)
        segmentation2 = Add()([segmentation2, segmentation1up])
        segmentation2 = UpSampling2D()(segmentation2)
        upsample4 = self.upsampling_module(localization3, self.filters)

        # Level 1
        concat4 = concatenate([upsample4, add1])
        conv6 = self.new_conv(concat4, self.filters * 2, self.kernel, 1)
        segmentation3 = self.segmentation_layer(conv6, self.filters)
        segmentation3 = Add()([segmentation3, segmentation2])

        # Output
        output = Conv2D(2, self.kernel, activation = "softmax", padding = self.padding)(segmentation3)
        model = Model(name = "ImprovedUnetModel", inputs = input_layer, outputs = output)
        self.model = model
        return
        