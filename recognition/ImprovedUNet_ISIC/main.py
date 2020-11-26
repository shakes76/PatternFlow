# Filename: main.py
# Author: Navin Sivasankaran
# Date created: 6/11/2020
# Date last modified (addressing feedback): 26/11/2020
# Python Version: 3.7.7
# Description: The main driver script, which is to be called. Contains everything, except for the model.

#Import libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pathlib
import glob
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.utils import shuffle
from model import *

#Set to GPU
devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(devices[0], True)

#Path (change the path as necessary)
images_source = pathlib.Path("D:\s4532664\COMP3710\ISIC2018_Task1-2_Training_Data")

#Input images
input_dir = images_source / "ISIC2018_Task1-2_Training_Input_x2\*.jpg"

#Segmentation images (label images)
seg_dir = images_source / "ISIC2018_Task1_Training_GroundTruth_x2\*.png"
                             
#Setup arrays for training, testing and validation datasets
X_train = None
X_test = None
X_valid = None
y_train = None
y_test = None
y_valid = None

training = None
validation = None
testing = None

"""
The following function will convert a passed-in filename to an image.
Processing, such as resizing, normalising (0 to 1) and casting (divide by 255) are done to it to be used in the model.

Parameters:
    input_file: File name of the input image (scan)
    seg_file: File name of the segmentation image (label image)

Returns:
    Tuple containing 2 tensors of the input image and segmentation image, after processing has been completed
"""
def convert_file_image(input_file, seg_file):
   
    input_image = tf.io.read_file(input_file)
    input_image = tf.image.decode_jpeg(input_image, channels = 1)
    input_image = tf.image.resize(input_image, [256, 256])
    input_image = tf.cast(input_image, tf.float32) / 255.0
    
    seg_image = tf.io.read_file(seg_file)
    seg_image = tf.image.decode_png(seg_image, channels = 1)
    seg_image = tf.image.resize(seg_image, [256, 256])
    seg_image = seg_image == [0, 255]
    
    return input_image, seg_image

"""
The following function creates TF datasets for training, testing and validation purposes, based on the data passed in.
The data will be split into the following: 3 datasets (Training - 50%, Validation - 25%, Testing - 25%)
The data will also be shuffled, before being split to allow for images of varying difficult to be potentially considered.

Parameters:
    inputdir: Path of directory that has the input images (scan)
    segdir: Path of directory that has the segmentation images (label)

Returns:
    Tuple containing 3 tensors of the training, validaton and testing datasets.
    Each of those datasets has both the input image and segmentation images.
"""
def load_data(inputdir, segdir):
    
    xtrain = sorted(glob.glob(str(inputdir)))
    ytrain = sorted(glob.glob(str(segdir)))
    
    #Shuffle the data
    xtrain, ytrain = shuffle(xtrain, ytrain)
    
    #Split the images into 3 datasets (Training - 50%, Validation - 25%, Testing - 25%)
    half_length = int(len(xtrain)/2)
    quarter_length_ceil = int(tf.math.ceil(len(xtrain)/4))
    
    global X_test, X_valid, X_train, y_test, y_valid, y_train
        
    X_test = xtrain[-(quarter_length_ceil-1):]
    X_valid = xtrain[half_length:half_length + quarter_length_ceil]
    X_train = xtrain[0:half_length]
    
    y_test = ytrain[-(quarter_length_ceil-1):]
    y_valid = ytrain[half_length:half_length + quarter_length_ceil]
    y_train = ytrain[0:half_length]
    
    #Create datasets
    traindata = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validdata = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    testdata = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    #Process every filename and convert to image
    traindata = traindata.map(convert_file_image)
    validdata = validdata.map(convert_file_image)
    testdata = testdata.map(convert_file_image)
    
    return traindata, validdata, testdata


def main():
    #Set no. of epochs to run model for
    EPOCHS = 5

    #Run the function of loading data into each dataset
    training, validation, testing = load_data(input_dir, seg_dir)

    #Retrieve model from model.py
    unet = unet_model()
    #Fit model onto training dataset with validation dataset as validation data
    results = unet.fit(x=training.batch(10), epochs = EPOCHS, validation_data = validation.batch(10))

    #Generate plots for the accuracy
    plt.plot(results.history['dice_coefficient'], label='Training data accuracy')
    plt.plot(results.history['val_dice_coefficient'], label = 'Test data accuracy')
    plt.title('Comparison of the training data and test data accuracies (DSC)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice coefficient')
    plt.legend(loc='lower right')

    #Generate plots for the loss
    plt.figure()
    plt.plot(results.history['loss'], label='Training data loss')
    plt.plot(results.history['val_loss'], label = 'Test data loss')
    plt.title('Comparison of the training data and test data losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    #Print out the test accuracy (DSC) and loss
    test_loss, test_acc = unet.evaluate(x=testing.batch(10))
    print("\n")
    print("The DSC on the testing dataset: ", test_acc)
    print("The loss on the testing dataset: ", test_loss)

    #Generate subplots to show comparisons between actual masks and prediction masks
    #5 samples x 3 images
    fig, ax = plt.subplots(nrows=5, ncols=3)
    row = 0

    for image, mask in testing.take(5):
        pred_mask = unet.predict(image[tf.newaxis,...])
        pred_mask = tf.argmax(pred_mask[0],axis=-1)
        mask = tf.cast(mask, tf.float32)

        disp_list = [tf.squeeze(image), tf.argmax(mask, axis=-1), pred_mask]

        for i in range(3):
            ax[row, i].imshow(disp_list[i], cmap = "gray")
            ax[row, i].axis('off')
        
        row += 1
    
    plt.show()

# Run the file
if __name__ == "__main__":
    main()
