#Contains the source code for training, validating, testing and saving the model

import imp
from modules import model
from dataset import *

import tensorflow as tf
from keras import backend as K
import joblib

path = "C:/Users/danie/Downloads/ISIC DATA/"

def train_model():
    #Load in data
    images, masks = load_data(path)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = train_test_valid(images, masks)

    #Define model
    improved_unet_model = model()

    
    #Use exponential decay as learning rate in form init*0.985^epoch
    learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_rate=0.985,
        decay_steps=12
    ) 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)  # type: ignore

    improved_unet_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[dice_similarity]
    )

    improved_unet_model.fit(
        train_x, 
        train_y,
        batch_size=32,
        epochs=50,
        validation_data=(valid_x, valid_y),
        steps_per_epoch=12
    )

    #Saving the file
    filename = "Improved_UNetModel.sav"
    joblib.dump(improved_unet_model, filename)

    return improved_unet_model

def dice_similarity(x, y):
    """
    Calculates the dice similarity
    Param: x - the predicated set of pixels
    Param: y - the set of ground truths
    Returns: (float) dice similarity

    **Assistance from [1]**
    """
    x_card = K.flatten(x)
    y_card = K.flatten(y)

    intersect = K.sum(x_card * y_card)
    #Note union is an inclusive i.e. includes intersection
    union = K.sum(x_card) + K.sum(y_card)

    return 2 * intersect / union

if __name__ == "__main__":
    print(train_model())
   