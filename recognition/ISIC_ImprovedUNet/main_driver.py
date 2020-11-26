"""
Author: Marko Uksanovic
Student Number: s4484509
"""
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

# Get a sorted list of all file paths
training_images = sorted(glob.glob(path_data + '/*.jpg'))
training_masks = sorted(glob.glob(path_labels + '/*.png'))

def resize(path):
    """
    Resize the images to 256 x 256

    Input:
        path(string): path to the image
    Returns:
        x(tensor): a tensor of shape [img_size * img_size]
    """
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=1)
    x = tf.image.resize(x,(img_size, img_size))

    return x

def normalise(image, mask):
    """
    Normalise the data to be between 0 and 1 to aid training

    Input:
        image(string): path to the image
        mask(string): path to the mask

    Returns:
        img(tensor): a tensor with 32bit floats as entries
        msk(tensor): a tensor with 32bit floats as entries
    """
    img = resize(image)
    img = tf.cast(img, tf.float32) / 255.0
    msk = resize(mask)
    msk = msk == [0, 255]
    msk = tf.cast(msk, tf.float32)

    return img, msk

def prepare_data(X, Y):
    """
    Puts the data into a form usable by the network and shuffles it

    Input: 
        X(tensor): a tensor with 32bit floats as entries
        Y(tensor): a tensor with 32bit floats as entries

    Returns:
        data(tensor): a tensor containing both X and Y tensor slices
    """
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.shuffle(len(X))
    data = data.map(normalise)

    return data

def dice_similarity(prediction, true):
    """
    The Sørensen–Dice coefficient, also known as the Dice similarity coefficient.
    Wikipedia gives the function as: coefficient = (2|X| * |Y|) / (|X| + |Y|)

    Input:
        prediction(tensor): the prediced image tensor
        true(tensor): the actual image tensor

    Returns:
        coefficient(float): the dice similarity coefficient
    """
    X = tf.keras.backend.flatten(true)
    Y = tf.keras.backend.flatten(prediction)

    num = (2.0 * tf.math.reduce_sum(X * Y))
    denom = tf.math.reduce_sum(X) + tf.math.reduce_sum(Y)
    coefficient = num / denom

    return coefficient

def test_predictions(data):
    """
    Compares the predicted and real masks

    Input:
        data(tensor): a tensor containing the predicted and actual mask

    Returns:
        predictions(list): a list containing the network predictions
        actual_masks(list): a list containing the actual masks
    """
    predictions = []
    actual_masks = []
    for img, mask in data:
        predicted_mask = model.predict(img[tf.newaxis,...])
        predictions.append(predicted_mask)
        actual_masks.append(mask)
    return predictions, actual_masks

dice_sim_coeff = dice_similarity

# Split the data into validation (80%) and testing (20%) sets
X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(training_images, training_masks, test_size = 0.2, random_state=7)
# Splot the validation set into a training (85%) and validation (15%) sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size = 0.15, random_state=7)

# Create the data tensors
Train = prepare_data(X_train, Y_train)
Valid = prepare_data(X_train_valid, Y_train_valid)
Test = prepare_data(X_test, Y_test)

# Generate the model
model = unet()
model.compile(optimizer = keras.optimizers.Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics=dice_sim_coeff)
model.fit(Train.batch(16), epochs=8, validation_data = Valid.batch(16))

# Make predicitons and test performance of testing set
predictions, truth = test_predictions(Test)
performance = dice_similarity(predictions, truth)

print("\nTest Dataset Dice similarity coefficient:" ,round(float(performance), 4)) 