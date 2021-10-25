"""
This script loads the input and output images from the ISIC dataset for pre-processing.
TODO: Add descriptions of future implementations here.
@author: Mujibul Islam Dipto
"""
import os # for operating system functionalities 
from sklearn.utils import shuffle, validation # for shuffling data 
import math # for mathematical operations
import tensorflow as tf # for DL functionalities 


print(tf. __version__) # check tf version
print("GPUs in use: ", len(tf.config.list_physical_devices('GPU'))) # check if tf has access to GPU

def main():
    """
    The main function that runs this script
    """
    # load data from ISIC dataset 
    images = sorted(item for item in os.listdir("../../../isic-data/ISIC2018_Task1-2_Training_Input_x2/") if item.endswith('jpg')) # training input
    masks = sorted(item for item in os.listdir("../../../isic-data/ISIC2018_Task1_Training_GroundTruth_x2/") if item.endswith('png')) # ground truth

    # suffle data before split to remove bias
    images, masks = shuffle(images, masks)
    total_length = len(images)
    """
    split data into 80% train, 10% validation and 10% test
    total legnth = 2594
    train_size = 2594 * 0.8 = 2074
    val_size = 260
    test_size = 260
    Note: Numbers were adjusted to fit the dataset
    """
    
    # data for train
    train_images = images[:2074]
    train_masks = masks[:2074]
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_masks))

    # data for validation
    val_images = images[2074:2074 + 260]
    val_masks = masks[2074:2074 + 260]
    val_data = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

    # data for test
    test_images = images[2074 + 260:]
    test_masks = masks[2074 + 260:]
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

# run main function
if __name__ == "__main__":
    main()