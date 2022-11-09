"""
Holds the Discriminator model for the GAN, see README.md for more information
on it's architecture. train.py imports this file.

@author Theo Duval
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense


class Discriminator(tf.keras.Model):
    """Holds the model for the discriminator part of the GAN."""

    # Define the optimiser for the discriminator to use
    optimiser = tf.keras.optimizers.SGD(learning_rate=5e-3)

    # Define the proportion for the dropout layers for easy adjustments
    dropout = 0.5


    def __init__(self):
        """
        Initialises all of the layers and inheritence for the discriminator.
        """

        # Set the padding for the convolutional layers
        self.padding = 'same'

        super(Discriminator, self).__init__()

        # The main convolutional layers with increasing filter size at
        # each layer.
        self.conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding=self.padding)
        self.bnorm1 = BatchNormalization()
        self.drop1 = Dropout(self.dropout)
        self.lrelu1 = LeakyReLU()

        self.conv2 = Conv2D(128, (3, 3), strides=(2, 2), padding=self.padding)
        self.bnorm2 = BatchNormalization()
        self.drop2 = Dropout(self.dropout)
        self.lrelu2 = LeakyReLU()

        self.conv3 = Conv2D(256, (3, 3), strides=(2, 2), padding=self.padding)
        self.bnorm3 = BatchNormalization()
        self.drop3 = Dropout(self.dropout)
        self.lrelu3 = LeakyReLU()

        self.conv4 = Conv2D(512, (3, 3), strides=(2, 2), padding=self.padding)
        self.bnorm4 = BatchNormalization()
        self.drop4 = Dropout(self.dropout)
        self.lrelu4 = LeakyReLU()

        # Now flatten and use Dense layers for the output
        self.flat = Flatten()
        self.dense1 = Dense(512)
        self.dropd1 = Dropout(self.dropout)
        self.lrelud1 = LeakyReLU()

        self.dense2 = Dense(1)


    def call(self, x):
        """Define a single call of this model."""

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.drop1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.drop2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.drop3(x)
        x = self.lrelu3(x)

        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.drop4(x)
        x = self.lrelu4(x)

        x = self.flat(x)
        x = self.dense1(x)
        x = self.dropd1(x)
        x = self.lrelud1(x)
        x = self.dense2(x)

        return x

