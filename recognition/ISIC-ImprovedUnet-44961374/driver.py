"""
This script loads the input and output images from the ISIC dataset for pre-processing.
TODO: Add descriptions of future implementations here.
@author: Mujibul Islam Dipto
"""
import os # for operating system functionalities 
from sklearn.utils import shuffle, validation # for shuffling data 
import math # for mathematical operations
import tensorflow as tf # for DL functionalities 
from tensorflow.python.client import device_lib

print(tf. __version__) # check tf version
print("GPUs in use: ", len(tf.config.list_physical_devices('GPU'))) # check if tf has access to GPU

def main():
    """
    The main function that runs this script
    """
    # load data from ISIC dataset 
    input_images = sorted(item for item in os.listdir("../../../isic-data/ISIC2018_Task1-2_Training_Input_x2/") if item.endswith('jpg')) # training input
    output_images = sorted(item for item in os.listdir("../../../isic-data/ISIC2018_Task1_Training_GroundTruth_x2/") if item.endswith('png')) # ground truth

    # suffle data before split to remove bias
    input_images, output_images = shuffle(input_images, output_images)
    total_length = len(input_images)
    """
    split data into 80% train, 10% validation and 10% test
    total legnth = 2594
    train_size = 2594 * 0.8 = 2074
    val_size = 260
    test_size = 260
    Note: Numbers were adjusted to fit the dataset
    """
    
    # data for train
    train_inputs = input_images[:2074]
    train_outputs = output_images[:2074]
    train_data = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs))

    # data for validation
    val_inputs = input_images[2074:2074 + 260]
    val_outputs = output_images[2074:2074 + 260]
    val_data = tf.data.Dataset.from_tensor_slices((val_inputs, val_outputs))

    # data for test
    test_inputs = input_images[2074 + 260:]
    test_outputs = output_images[2074 + 260:]
    test_data = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs))
    print(device_lib.list_local_devices())

# run main function
if __name__ == "__main__":
    main()