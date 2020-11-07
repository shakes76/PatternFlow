'''
Build, train and test a TensorFlow based model to segment ISICs 2018 dermatology data
'''

# Import modules
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from math import ceil
from glob import glob
from IUNet_sequence import *
from IUNet_model import *

# Define global variables
TRAIN_SIZE = 0.8
VALIDATE_SIZE = 0.1
TRAIN_BATCH_SIZE = 10
VALIDATE_BATCH_SIZE = 10

# Allow GPU more memory access
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def train_model(x_data_path, y_data_path, x_file_ext, y_file_ext, num_epochs, batch_size):
    # Import data
    train_seq, validate_seq, test_seq = split_data(x_data_path, y_data_path, x_file_ext, y_file_ext, TRAIN_SIZE, VALIDATE_SIZE, batch_size)

    # Create model
    model = iunet_model()
    #print(model.summary())

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=dice_coef_loss,
        metrics=['accuracy'])

    print('---- TRAINING MODEL ----')
    # Train model
    history = model.fit(
        train_seq,
        epochs=num_epochs,
        validation_data=validate_seq,
    )

    # Evaluate model
    print('---- EVALUATING MODEL ON TEST DATA ----')
    model.evaluate(test_seq)

    # Plot the loss and accuracy curves of the training
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label ='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='upper left')
    plt.show()
