
from typing import Concatenate
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Conv2DTranspose

class Unet(tf.keras.Model):
    def __init__(self):
        super(Unet, self).__init__()
        self.cat = 0
        self.conv_args = {
        "activation": "relu",
        "kernel_initializer": "HeNormal",
        "padding": "same",
        }
        self.skip_connection_1 = 0
        self.skip_connection_2 = 0
        self.skip_connection_3 = 0
        self.skip_connection_4 = 0


    def contraction(self, input):
        x = Conv2D(64, 3, **self.conv_args)(input)
        x = Conv2D(64, 3, **self.conv_args)(x)
        x = BatchNormalization()(x, training=False)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        self.skip_connection_1 = x

        x = Conv2D(128, 3, **self.conv_args)(x)
        x = Conv2D(128, 3, **self.conv_args)(x)
        x = BatchNormalization()(x, training=False)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        self.skip_connection_2 = x

        x = Conv2D(256, 3, **self.conv_args)(x)
        x = Conv2D(256, 3, **self.conv_args)(x)
        x = BatchNormalization()(x, training=False)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        self.skip_connection_3 = x

        x = Conv2D(512, 3, **self.conv_args)(x)
        x = Conv2D(512, 3, **self.conv_args)(x)
        x = BatchNormalization()(x, training=False)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = MaxPool2D(pool_size = (2,2))(x)
        self.skip_connection_4 = output
        return output

    def bottleneck(self, input):
        x = Conv2D(1024, 3, **self.conv_args)(input)
        output = Conv2D(1024, 3, **self.conv_args)(x)
        return output

    def expansion(self, input):
        x = Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(input)
        x = Concatenate([x, self.skip_connection_1], axis=3)
        x = Conv2D(512, 3, **self.conv_args)(x)
        x = Conv2D(512, 3, **self.conv_args)(x)

        x = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(input)
        x = Concatenate([x, self.skip_connection_2], axis=3)
        x = Conv2D(256, 3, **self.conv_args)(x)
        x = Conv2D(256, 3, **self.conv_args)(x)

        x = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(input)
        x = Concatenate([x, self.skip_connection_3], axis=3)
        x = Conv2D(128, 3, **self.conv_args)(x)
        x = Conv2D(128, 3, **self.conv_args)(x)

        x = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(input)
        x = Concatenate([x, self.skip_connection_4], axis=3)
        x = Conv2D(64, 3, **self.conv_args)(x)
        output = Conv2D(64, 3, **self.conv_args)(x)
        return output

    def output_layer(self, input):
        output = Conv2D(3, 1, padding="same", activation = "softmax")(input)
        return output
    
    def call(self, input):
        x = self.contraction(input)
        x = self.bottleneck(x)
        x = self.expansion(x)
        output = self.output_layer(x)
        return output
    

    

