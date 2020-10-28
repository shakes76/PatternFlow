"""
Laterality classification of the OAI AKOA knee data set.

@author Jonathan Godbold, s4533974.

Usage of this file is strictly for The University of Queensland.
Date: 27/10/2020.

Description:
Builds a model of the OASIS OKOA dataset.
"""

# Import libraries.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Print the current version.
print('TensorFlow version:', tf.__version__)

def addLayer(model, input_shape, weight_decay, n_filters, kernel_size, padding, kernel_regularizer, batch_norm, activation_func):
    """
    Adds a convolutional 2D layer to current model.
    Format returned: model to be trained.
    Paramters:
    model - model to add layer to.
    input_shape - shape of input image.
    weight_decay - learning rate.
    n_filters - number of filters in the convolutional layer.
    kernel_size - size of kernel.
    padding - type.
    kernel_regularizer - L2 or L1.
    batch_norm - true if batch is normalized, false otherwise.
    activation_func - Normally ReLu or Sigmoid activation.
    """
    model.add(Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    if (batch_norm == True):
        model.add(BatchNormalization())
    model.add(Activation(activation_func))
    return model

def buildNetwork(train_images):
    """
    Builds a network given the specified parameters.
    Format returned: model to be trained.
    """
    model = Sequential()
    shape = train_images
    weight_decay = 1e-4
    k_size = (3, 3)
    reg = regularizers.l2(weight_decay)
    model = addLayer(model, shape, weight_decay, 32, k_size, "same", reg, True, 'relu')
    model = addLayer(model, shape, weight_decay, 64, k_size, "same", reg, True, 'relu')
    model = addLayer(model, shape, weight_decay, 128, k_size, "same", reg, True, 'relu')
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


class MyThresholdCallback(tf.keras.callbacks.Callback):
    """
    Threshold function to stop training when a specific threshold of
    validation accuracy has been reached to prevent overfitting.
    This code is not my own. Source: https://stackoverflow.com/questions/59563085/how-to-stop-training-when-it-hits-a-specific-validation-accuracy
    Author: StackOverflow user sebastian-ez.
    Code could be removed since this threshold was only reached on the 5th (last) epoch.
    """
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True

def compile_and_run(model, epochs, batch):
    """
    Compiles and runs the model.
    - Uses Adam optimizer.
    - Loss function is binary_crossentropy.
    """
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    my_callback = MyThresholdCallback(threshold=0.9)
    history = model.fit(train_images, train_images_y, epochs, validation_data=(validate_images, validate_images_y), callbacks=[my_callback], batch_size = batch)

print("Model successfully built and tested. Application exiting...")