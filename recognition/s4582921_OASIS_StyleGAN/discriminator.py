import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

KERNEL_SIZE = 3
DROPOUT = 0.2
MOMENTUM = 0.8
BETA = 0.999
ALPHA = 0.2

class Discriminator():


    def __init__(self, image_size, blocks, learning_rate, channels):
        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)
        self.momentum = MOMENTUM
        self.dropout = DROPOUT

        self.image_size = image_size
        self.blocks = blocks
        self.channels = channels

        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, 0, BETA)


    def train_block(self, input, filters):
        resolution = Conv2D(filters=filters, kernel_size=1)(input)

        output = Conv2D(filters=filters, kernel_size=self.kernel_size, padding='same')(input)
        output = LeakyReLU(ALPHA)(output)
        output = Conv2D(filters=filters, kernel_size=self.kernel_size, padding='same')(output)
        output = LeakyReLU(ALPHA)(output)

        output = add([resolution, output])
        return output


    def build_model(self):

        input = Input(shape=[self.image_size, self.image_size, 1])
        output = input

        for block in range(self.blocks):
            output = self.train_block(output, self.channels * 2**block)

        output = AveragePooling2D()(output)
        output = Flatten()(output)
        output = Dense(1, activation='sigmoid')(output)

        return Model(inputs=input, outputs=output)


    def loss(self, real_output, fake_output):

        return tf.keras.losses.BinaryCrossentropy(tf.ones_like(real_output), real_output) + tf.keras.losses.BinaryCrossentropy(tf.zeros_like(fake_output), fake_output)

    def w_loss(self, real_output, fake_output):

        return tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)







