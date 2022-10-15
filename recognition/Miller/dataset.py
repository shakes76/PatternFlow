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

    # Create array of arrays
    train_set = np.array(image_list, dtype=np.float32)
    return train_set

# Normalizes training images and adds 4th dimention 
def process_training (data_set):

    # Calculate the residuals of the data - each residual is dist from each distribution mean which is now zero
    data_set = (data_set - np.mean(data_set)) / np.std(data_set)
    # Rescale Data - ratio of dist of each value from min value in each dataset to range of values in each dataset -> value between (0,1) now
    # Forces dataset to be same scale, and perseves shape of distribution -> "Squeezed and shifted to fit between 0 and 1"
    data_set= (data_set - np.amin(data_set)) / np.amax(data_set - np.amin(data_set))
    # Add 4th dimension
    data_set = data_set [:,:,:,np.newaxis]
    
    return data_set

"""
# loads labels images and map pixel values to class indices and convert image data type to unit8 
def load_labels (path):
    image_list =[]

    for filename in glob.glob(path+'/*.png'): 
        im=image.imread (filename)
        one_hot = np.zeros((im.shape[0], im.shape[1]))
        for i, unique_value in enumerate(np.unique(im)):
          one_hot[:, :][im == unique_value] = i
        image_list.append(one_hot)

    print('train_y shape:',np.array(image_list).shape)
    labels = np.array(image_list, dtype=np.uint8)
    
    pyplot.imshow(labels[2])
    pyplot.show()

    return labels

# one hot encode label data and convert to numpy array
def process_labels(seg_data):
    onehot_Y = []
    for n in range(seg_data.shape[0]): 
      im = seg_data[n]
      n_classes = 4
      one_hot = np.zeros((im.shape[0], im.shape[1], n_classes),dtype=np.uint8)
      for i, unique_value in enumerate(np.unique(im)):
          one_hot[:, :, i][im == unique_value] = 1
      onehot_Y.append(one_hot)
    
    onehot_Y =np.array(onehot_Y)
    print (onehot_Y.dtype)
    #print (np.unique(onehot_validate_Y))
    print (onehot_Y.shape)

    return onehot_Y"""