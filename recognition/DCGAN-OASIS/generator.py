"""
Holds the Generator model for the GAN, see README.md for more information on
it's architecture. train.py imports this file.

@author Theo Duval
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dropout


class Generator(tf.keras.Model):
    """A subclass of tf.keras.Model that is the generator of the GAN."""

    # Define the optimiser to use for the generator
    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Define the proportion for the dropout layers for easy adjustments
    dropout = 0


    def __init__(self):
        """Initialises all of the layers and inheritence for the generator."""

        super(Generator, self).__init__()

        # Set the bias and padding
        self.bias = False
        self.padding='same'


        # The input is an array of 100 random numbers from a Gaussian
        # distribution. The number of filters is based on the reduced
        # resolution.
        self.dense = Dense(4*4*512, use_bias=self.bias)

        # Use BatchNormalization to reduce impacts of poor initialisation
        self.bnorm1 = BatchNormalization()

        # Use LeakyReLU to help reduce killing neruons in cases of low
        # confidence.
        self.lrelu1 = LeakyReLU()


        # Reshape from 1D to the reduced image dimensions
        self.reshape = Reshape((4, 4, 512))


        # The main convolutional layers
        self.conv2 = Conv2DTranspose(512, (3, 3), strides=(2, 2), 
                                     padding=self.padding, use_bias=self.bias)
        self.bnorm2 = BatchNormalization()
        self.lrelu2 = LeakyReLU()
        self.drop2 = Dropout(self.dropout)

        self.conv3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), 
                                     padding=self.padding, use_bias=self.bias)
        self.bnorm3 = BatchNormalization()
        self.lrelu3 = LeakyReLU()
        self.drop3 = Dropout(self.dropout)

        self.conv4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), 
                                     padding=self.padding, use_bias=self.bias)
        self.bnorm4 = BatchNormalization()
        self.lrelu4 = LeakyReLU()
        self.drop4 = Dropout(self.dropout)
        
        self.conv5 = Conv2DTranspose(64, (3, 3), strides=(2, 2), 
                                     padding=self.padding, use_bias=self.bias)
        self.bnorm5 = BatchNormalization()
        self.lrelu5 = LeakyReLU()
        self.drop5 = Dropout(self.dropout)

        # Use sigmoid activation for final layer
        self.conv6 = Conv2DTranspose(1, (3, 3), strides=(1, 1),
                                     activation='sigmoid', 
                                     padding=self.padding, 
                                     use_bias=self.bias)


    def call(self, x):
        """The definition of a single call of this model"""
        x = self.dense(x)
        x = self.bnorm1(x)
        x = self.lrelu1(x)

        x = self.reshape(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.lrelu2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.lrelu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.lrelu4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.bnorm5(x)
        x = self.lrelu5(x)
        x = self.drop5(x)

        x = self.conv6(x)

        return x