"""
This script loads the input and output images from the ISIC dataset and performs pre-processing.
TODO: Add descriptions of future implementations here.
@author: Mujibul Islam Dipto
"""
import os # for operating system functionalities 
from sklearn.utils import shuffle, validation # for shuffling data 
import tensorflow as tf # for DL functionalities 
from process_data import process_data # for dataset processing functionalities
import matplotlib.pyplot as plt # for plotting images
from data_loader import load_data # for loading the datasets
from model import create_model # for creating the improved unet model

print(tf. __version__) # check tf version
print("GPUs in use: ", len(tf.config.list_physical_devices('GPU'))) # check if tf has access to GPU


def main():
    """
    The main function that runs this script
    """
    # load processed data using data_loader and process_data modules
    train_data, val_data, test_data = load_data()
    # create an improved unet model
    improved_unet_model = create_model(2)

    # training parametets
    EPOCHS = 2
    BATCH_SIZE = 16
    improved_unet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = improved_unet_model.fit(train_data, steps_per_epoch=BATCH_SIZE, epochs=EPOCHS, validation_data=val_data)

# run main function
if __name__ == "__main__":
    main()