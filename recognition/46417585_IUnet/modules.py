import tensorflow as tf
import keras

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

# Could need some skips in here
inputs = Input(shape=(256, 256, 1))
conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool3)
up1 = UpSampling2D(size=(2, 2))(conv4)
convT1 = Conv2DTranspose(128, (3, 3), padding="same", activation="relu")(up1)
up2 = UpSampling2D(size=(2, 2))(convT1)
convT2 = Conv2DTranspose(128, (3, 3), padding="same", activation="relu")(up2)
up3 = UpSampling2D(size=(2, 2))(convT2)
convT3 = Conv2DTranspose(128, (3, 3), padding="same", activation="relu")(up3)
up4 = UpSampling2D(size=(2, 2))(convT3)
convT4 = Conv2DTranspose(128, (3, 3), padding="same", activation="relu")(up4)
