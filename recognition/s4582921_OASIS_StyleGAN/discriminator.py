"""
discriminator.py

The file containing the discriminator class and its parameters.

Requirements:
    - tensorflow-gpu - 2.4.1
    - matplotlib - 3.4.3

Author: Bobby Melhem
Python Version: 3.9.7
"""


import tensorflow as tf
from tensorflow.keras.backend import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


#Hyper Parameters
KERNEL_SIZE = 3
BETA1 = 0
BETA2 = 0.999
ALPHA = 0.2


class Discriminator():
    """
    The discriminator model which trains to identify real or fake images.

    Attributes:
        image_size : the size of images being classified
        blocks : number of train blocks used in building the model
        learning_rate : learning rate of the optimizer
        channels : channels in image used for training
        kernel_size : size of the kernel in convolutional layers
        model : reference to the discriminator model
        optimizer : optimizer used in the discriminator model
    """


    def __init__(self, image_size, blocks, learning_rate, channels):
        """Initialises an instance of the Discriminator"""

        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)

        self.image_size = image_size
        self.blocks = blocks
        self.channels = channels

        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, BETA1, BETA2)


    def train_block(self, input, filters):
        """
        A training block used in building the model.

        Args:
            input : previous layer as input
            filters : number of filters to use in convolution layers
        Returns:
            The output layer produced by the block.
        """

        resolution = Conv2D(filters=filters, kernel_size=1)(input)

        output = Conv2D(filters=filters, kernel_size=self.kernel_size, padding='same')(input)
        output = LeakyReLU(ALPHA)(output)
        output = Conv2D(filters=filters, kernel_size=self.kernel_size, padding='same')(output)
        output = LeakyReLU(ALPHA)(output)

        output = add([resolution, output])

        return output


    def build_model(self):
        """
        Builds the discriminator model.

        Returns:
            The model for the discriminator.
        """

        input = Input(shape=[self.image_size, self.image_size, 1])
        output = input

        for block in range(self.blocks):
            output = self.train_block(output, self.channels * 2**block)

        output = AveragePooling2D()(output)
        output = Flatten()(output)
        output = Dense(1, activation='sigmoid')(output)

        return Model(inputs=input, outputs=output)


    def loss(self, real_output, fake_output):
        """
        Basic binary cross entropy loss.

        Args:
            real_output : classification of a batch of real images
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        criterion = tf.keras.losses.BinaryCrossentropy()

        return criterion(tf.ones_like(real_output), real_output) + criterion(tf.zeros_like(fake_output), fake_output) / 2


    def l_loss(self, real_output, fake_output):
        """
        Logistic loss used in official implementation.

        Args:
            real_output : classification of a batch of real images
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        return -tf.nn.softplus(fake_output) + tf.nn.softplus(-real_output)


    def w_loss(self, real_output, fake_output):
        """
        Wasserstein loss used in official implementation.

        Args:
            real_output : classification of a batch of real images
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        return  tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


    def h_loss(self, real_output, fake_output):
        """
        Hinge loss.

        Args:
            real_output : classification of a batch of real images
            fake_output : classification of a batch of fake images
        Returns:
            The loss.
        """

        return  mean(relu(1 + real_output) + relu(1 - fake_output))






