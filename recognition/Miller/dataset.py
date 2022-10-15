"""
dataset.py" containing the data loader for loading and preprocessing your data
"""

import tensorflow as tf
import glob
import numpy as np
from matplotlib import pyplot
from matplotlib import image

# Download the Oasis Data as zip file. Will need to extract it manually afterwards
def download_oasis ():
    
    dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
    # Download file from URL Path, origin=path, fname=file name, untar=compress file
    tf.keras.utils.get_file(origin=dataset_url,fname='oa-sis' ,untar=True)
  

# Loads the training images (non segmented) from given path and returns an numpy array of arrays
def load_training (path):
    image_list = []

    # Iterate through all paths and convert to 'png'
    for filename in glob.glob(path + '/*.png'): 
        # Read an image from the given filename into an array
        im = image.imread (filename)
        # Append array to list
        image_list.append(im)

    print('train_X shape:', np.array(image_list).shape)

    # Create an numpy array to hold all the array turned images
    train_set = np.array(image_list, dtype=np.float32)
    return train_set

# Normalizes training images and adds 4th dimention 
def process_training (data_set):

    """ Residual Extraction -> Useful for comparing distributions with different means but similar shapes"""
    # Calculate the residuals of the data - each residual is dist from each distribution mean which is now zero
    data_set = (data_set - np.mean(data_set)) / np.std(data_set)
    """ Min-Max Rescaling -> Useful for comparign distributions with different scales or different shapes"""
    # Rescale Data - ratio of dist of each value from min value in each dataset to range of values in each dataset -> value between (0,1) now
    # Forces dataset to be same scale, and perseves shape of distribution -> "Squeezed and shifted to fit between 0 and 1"
    data_set= (data_set - np.amin(data_set)) / np.amax(data_set - np.amin(data_set))
    # Add 4th dimension
    data_set = data_set [:,:,:,np.newaxis]
    
    return data_set

# Loads labels images from given path and map pixel values to class indices and convert image data type to unit8 
def load_labels (path):
    image_list =[]

    # Iterate through all paths and convert to 'png'
    for filename in glob.glob(path+'/*.png'): 
        # Read an image from the given filename into an array
        im=image.imread (filename)
        # Create 'im.shape[0] x im.shape[1]' shaped array of arrays of zeros
        one_hot = np.zeros((im.shape[0], im.shape[1]))
        # Iterate through sorted and unique arrays of given array turned image
        for i, unique_value in enumerate(np.unique(im)):
            # One hot each unique array with its numerical value of its entry in the dataset -> transform categorical into numerical dummy features
            one_hot[:, :][im == unique_value] = i
        # Append array to list
        image_list.append(one_hot)

    print('train_y shape:',np.array(image_list).shape)

    # Create an numpy array to hold all the array turned images
    labels = np.array(image_list, dtype=np.uint8)
    
    #pyplot.imshow(labels[2])
    #pyplot.show()

    return labels

# One hot encode label data and convert to numpy array
def process_labels(seg_data):
    onehot_Y = []

    # Iterate through all array turned images by shapes first value
    for n in range(seg_data.shape[0]): 
        
        # Get data at position in array
        im = seg_data[n]

        # There are 4 classes
        n_classes = 4

        # Create 'im.shape[0] x im.shape[1] x n_classes' shaped array of arrays of arrays of zeros with type uint8
        one_hot = np.zeros((im.shape[0], im.shape[1], n_classes),dtype=np.uint8)
    
        # Iterate through sorted and unique arrays of given array turned image
        for i, unique_value in enumerate(np.unique(im)):
            # One hot each unique array with its numerical value of its entry in the dataset -> transform categorical into numerical dummy features
            one_hot[:, :, i][im == unique_value] = 1
            # Append array to list
            onehot_Y.append(one_hot)
    
    # Create an numpy array to hold all the array turned images
    onehot_Y =np.array(onehot_Y)
    #print (onehot_Y.dtype)
    #print (np.unique(onehot_validate_Y))
    #print (onehot_Y.shape)

    return onehot_Y