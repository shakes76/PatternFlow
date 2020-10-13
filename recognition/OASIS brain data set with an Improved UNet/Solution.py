# import necessary modules
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def solution(h, w, n_classes):
    """
    Creates and return the UNET model to be used by the driver_script.py script
    
    This is where I will implement my solution to the problem.
    Must only use tensorflow
    My function simply creates and compiles the UNET model to be used by
    the driver_script.py file.
    """
    
    # create model
    model = models.Sequential()
    
    # add convolution layers    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    # add dense layers
    model.add(layers.Flatten(input_shape=(h, w, 1)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    
    # compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    # return the model
    return model
    