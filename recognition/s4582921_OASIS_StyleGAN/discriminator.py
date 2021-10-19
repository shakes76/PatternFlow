import tensorflow as tf
from keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense

KERNEL_SIZE = 3
DROPOUT = 0.2
MOMENTUM = 0.8

class Discriminator():


    def __init__(self, image_size, learning_rate, beta):
        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)
        self.momentum = MOMENTUM
        self.dropout = DROPOUT

        self.image_size = image_size

        self.model = tf.keras.models.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta)
        self.criterion = tf.keras.losses.BinaryCrossentropy()


    def train_block(self, filters):
        self.model.add(Conv2D(filters=filters, kernel_size=self.kernel_size, activation='relu', padding='same'))
        self.model.add(Dropout(self.dropout))
        self.model.add(BatchNormalization(momentum=self.momentum))


    def build_model(self):

        for power in range(-1, 2):
            self.train_block(1/(2**power) * self.image_size)

        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))


    def loss(self, real_output, fake_output):
        return self.criterion(tf.ones_like(real_output), real_output) + self.criterion(tf.zeros_like(fake_output), fake_output)







