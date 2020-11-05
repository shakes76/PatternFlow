import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tensorflow import keras
from sklearn.model_selection import train_test_split
from model import unet

# Import the data
path_data = "C:/Users/Marko/Desktop/data"
path_labels = "C:/Users/Marko/Desktop/masks"
img_size = 256

def convert(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=1)
    x = tf.image.resize(x,(img_size, img_size))

    return x

def normalise(image, mask):
    img = convert(image)
    img = tf.cast(img, tf.float32) / 255.0
    msk = convert(mask)
    msk = msk == [0, 255]
    msk = tf.cast(msk, tf.float32)

    return img, msk

def prepare_data(X, Y):
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.shuffle(len(X))
    data = data.map(normalise)

    return data

training_images = sorted(glob.glob(path_data + '/*.jpg'))
training_masks = sorted(glob.glob(path_labels + '/*.png'))

X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(training_images, training_masks, test_size = 0.2, random_state=7)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size = 0.15, random_state=7)

Train = prepare_data(X_train, Y_train)
Valid = prepare_data(X_train_valid, Y_train_valid)
Test = prepare_data(X_test, Y_test)

model = unet()
model.compile(optimizer = keras.optimizers.Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics=dice_coef)
model.fit(Train.batch(16), epochs=10, validation_data = Valid.batch(16))