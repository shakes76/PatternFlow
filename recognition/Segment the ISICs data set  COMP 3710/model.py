import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Unet(keras.Model):
    def __init__(self):
        super(Unet, self).__init__()
        # 1
        self.conv1_1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv1_1")
        self.conv1_2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv1_2")
        ## max pool
        self.down_pool1 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool1")

        # 2
        self.conv2_1 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2_1")
        self.conv2_2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2_2")
        self.down_pool2 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool2")

        # 3
        self.conv3_1 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv3_1")
        self.conv3_2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv3_2")
        self.down_pool3 = layers.MaxPool2D(pool_size=(2, 2), name="down_pool3")


        self.conv9_1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_1")
        self.conv9_2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_2")
        # conv 1x1
        self.conv9_3 = layers.Conv2D(2, 1, activation='relu', padding='same', kernel_initializer='he_normal', name="conv9_3")
        # out
        self.out = layers.Conv2D(1, 1, activation='sigmoid', name="out")

    def call(self, input, training=False):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        down_pool1 = self.down_pool1(conv1_2)

        conv2_1 = self.conv2_1(down_pool1)
        conv2_2 = self.conv2_2(conv2_1)
        down_pool2 = self.down_pool2(conv2_2)

        conv3_1 = self.conv3_1(down_pool2)
        conv3_2 = self.conv3_2(conv3_1)
        down_pool3 = self.down_pool3(conv3_2)

        out = self.out(conv9_3)
        return out
















