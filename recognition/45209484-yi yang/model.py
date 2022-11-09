import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model

# build model
def model():
    input_layer = tf.keras.layers.Input(shape=(256,256,3))

    x = Conv2D(64,(3,3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(64,(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x1)
    x = Conv2D(128,(3,3), activation='relu', padding='same')(x)
    x2 = Conv2D(128,(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x2)
    x = Conv2D(256,(3,3), activation='relu', padding='same')(x)
    x3 = Conv2D(256,(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x3)
    x = Conv2D(512,(3,3), activation='relu', padding='same')(x)
    x4 = Conv2D(512,(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(1024,(3,3), activation='relu', padding='same')(x)
    encoded = Conv2D(1024,(3,3), activation='relu', padding='same')(x)

    u4 = UpSampling2D((2,2))(encoded)
    x = concatenate([x4, u4])
    x = Conv2D(1024,(3,3), activation='relu', padding='same')(x)
    x = Conv2D(512,(3,3), activation='relu', padding='same')(x)
    u3 = UpSampling2D((2,2))(x)
    x = concatenate([x3, u3])
    x = Conv2D(512,(3,3), activation='relu', padding='same')(x)
    x = Conv2D(256,(3,3), activation='relu', padding='same')(x)
    u2 = UpSampling2D((2,2))(x)
    x = concatenate([x2, u2])
    x = Conv2D(256,(3,3), activation='relu', padding='same')(x)
    x = Conv2D(128,(3,3), activation='relu', padding='same')(x)
    u1 = UpSampling2D((2,2))(x)
    x = concatenate([x1, u1])
    x = Conv2D(128,(3,3), activation='relu', padding='same')(x)
    x = Conv2D(64,(3,3), activation='relu', padding='same')(x)
    x = Conv2D(64,(3,3), activation='relu', padding='same')(x)

    decoded = Conv2D(1,(1,1), activation='sigmoid')(x)

    autoencoder = Model(input_layer, decoded)
    return autoencoder
