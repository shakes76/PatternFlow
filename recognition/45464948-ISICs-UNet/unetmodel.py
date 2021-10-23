import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D


def model():
    #encoder part
    input_layer = tf.keras.layers.Input(shape=(256,256,3))

    conv1 = Conv2D(64,(3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(64,(3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), padding='same')(conv1)
    conv2 = Conv2D(128,(3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128,(3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), padding='same')(conv2)
    conv3 = Conv2D(256,(3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256,(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), padding='same')(conv3)
    conv4 = Conv2D(512,(3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512,(3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), padding='same')(conv4)
    conv5 = Conv2D(1024,(3,3), activation='relu', padding='same')(pool4)
    encoded = Conv2D(1024,(3,3), activation='relu', padding='same')(conv5)