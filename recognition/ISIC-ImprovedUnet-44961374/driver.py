"""
This script loads the input and output images from the ISIC dataset for pre-processing.
TODO: Add descriptions of future implementations here.
@author: Mujibul Islam Dipto
"""
import os # for operating system functionalities 
from sklearn.utils import shuffle, validation # for shuffling data 
import math # for mathematical operations
import tensorflow as tf # for DL functionalities 
from process_data import process_data # for dataset processing functionalities
import matplotlib.pyplot as plt # for plotting images

print(tf. __version__) # check tf version
print("GPUs in use: ", len(tf.config.list_physical_devices('GPU'))) # check if tf has access to GPU

IMG_PATH = "../../../isic-data/ISIC2018_Task1-2_Training_Input_x2/" # path of images (scans) from the dataset
MASK_PATH = "../../../isic-data/ISIC2018_Task1_Training_GroundTruth_x2/" # path of mask images from the dataset


def main():
    """
    The main function that runs this script
    """
    # load data from ISIC dataset 
    images = sorted(os.path.join(IMG_PATH, item) for item in os.listdir(IMG_PATH) if item.endswith('jpg')) # training input
    masks = sorted(os.path.join(MASK_PATH, item) for item in os.listdir(MASK_PATH) if item.endswith('png')) # ground truth

    # suffle data before split to remove bias   
    images, masks = shuffle(images, masks)
    total_length = len(images)

    # split data into 80% train, 10% validation and 10% test
    # data for train
    train_images = images[:2074]
    train_masks = masks[:2074]
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    train_data = train_data.map(process_data)

    # data for validation
    val_images = images[2074:2074 + 260]
    val_masks = masks[2074:2074 + 260]
    val_data = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    val_data = val_data.map(process_data)

    # data for test
    test_images = images[2074 + 260:]
    test_masks = masks[2074 + 260:]
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    test_data = test_data.map(process_data)

# run main function
if __name__ == "__main__":
    main()