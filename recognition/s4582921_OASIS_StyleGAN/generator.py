import tensorflow as tf
from keras.layers import Dense, Reshape, UpSampling2D, BatchNormalization, Conv2D

KERNEL_SIZE = 3
MOMENTUM = 0.8

class Generator():


    def __init__(self, image_size, learning_rate, beta, blocks, channels):
        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)
        self.momentum = MOMENTUM

        self.image_size = image_size
        self.blocks = blocks
        self.channels = channels

        self.model = tf.keras.models.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta)
        self.criterion = tf.keras.losses.BinaryCrossentropy()


    def train_block(self, filters):
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(filters=filters, kernel_size=self.kernel_size, activation='relu', padding='same'))
        self.model.add(BatchNormalization(momentum=self.momentum))


    def build_model(self):
        #blocks = 4
        scale = 2**(self.blocks - 1)

        self.model.add(Dense(4 * 4 * scale * self.image_size))
        self.model.add(Reshape([4, 4, scale * self.image_size]))

        for factor in range(self.blocks):
            self.train_block((scale / 2**factor) * self.image_size)

        self.model.add(Conv2D(filters=self.channels, kernel_size=self.kernel_size, activation='tanh', padding='same'))


    def loss(self, fake_output):
        return self.criterion(tf.ones_like(fake_output), fake_output)






