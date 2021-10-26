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
from data_loader import load_data

print(tf. __version__) # check tf version
print("GPUs in use: ", len(tf.config.list_physical_devices('GPU'))) # check if tf has access to GPU


def main():
    """
    The main function that runs this script
    """
    # load processed data using data_loader and process_data modules
    train_data, val_data, test_data = load_data()


# run main function
if __name__ == "__main__":
    main()