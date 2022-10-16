from dataset import get_data
from modules import create_model
import tensorflow as tf
import numpy as np


def model_train():
    """

    :return: get the trained model of Improved-Unet
    """

    # to get the data required for training
    # minus the mean of data to get an input with 0 mean
    x_train = get_data("/Users/skywu/COMP3710-report/dataset/train_data/")
    x_train -= np.mean(x_train)
    y_train = get_data("/Users/skywu/COMP3710-report/dataset/train_truth/", is_y=True)
    x_validation = get_data("/Users/skywu/COMP3710-report/dataset/validation_data/")
    x_validation -= np.mean(x_validation)
    y_validation = get_data("/Users/skywu/COMP3710-report/dataset/validation_truth/", is_y=True)

    # create model
    Improved_unet = create_model()

    # define learning schedule with a decay rate of 0.985 each epoch
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=11,
        decay_rate=0.985
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile the model
    Improved_unet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    # model training
    Improved_unet.fit(x_train, y_train,
                      batch_size=100,
                      epochs=300,
                      shuffle=True,
                      validation_data=(x_validation, y_validation),
                      steps_per_epoch=11
                      )
    return Improved_unet
