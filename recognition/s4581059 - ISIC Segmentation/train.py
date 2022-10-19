#Contains the source code for training, validating, testing and saving the model

from modules import model
from dataset import *

import tensorflow as tf
import numpy as np

path = "C:/Users/danie/Downloads/ISIC DATA/"

def train():
    #Load in data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    
    #Process Data
    train_x = decode_image(train_x)
    valid_x = decode_image(valid_x)
    train_y = decode_mask(train_y)
    valid_y = decode_mask(valid_y)
    
    #Define model
    improved_unet_model = model()

    batch_size = 2
    number_batches = 100
    no_epochs = 300
    
    #Use exponential decay as learning rate in form init*0.985^epoch
    learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_rate=0.985,
        decay_steps=12
    ) 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)  # type: ignore

    improved_unet_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    improved_unet_model.fit(
        train_x, 
        train_y,
        batch_size=100,
        epochs=20,
        validation_data=(valid_x, valid_y),
        steps_per_epoch=12
    )
    
    return improved_unet_model
   