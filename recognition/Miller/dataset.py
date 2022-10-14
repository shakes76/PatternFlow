"""
dataset.py" containing the data loader for loading and preprocessing your data
"""
import tensorflow as tf
import pathlib
import glob
import numpy as np
from matplotlib import pyplot
from matplotlib import image

# Load Data and Process it
# Download the Oasis Data
def download_oasis ():
    
    dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,fname='oa-sis' ,untar=True)
    data_dir = pathlib.Path(data_dir)
    data_dir = data_dir

    # unzip data to current directory 
    #! unzip /root/.keras/datasets/oa-sis.tar.gz

# Loads the training images (non segmented) in the path and store in numpy array
def load_training (path):
    image_list = []

    # Iterate through all paths and convert to 'png'
    for filename in glob.glob(path + '/*.png'): 
        im=image.imread (filename)
        image_list.append(im)

    print('train_X shape:',np.array(image_list).shape)
    train_set = np.array(image_list, dtype=np.float32)
    return train_set
"""
# Normalizes training images and adds 4th dimention 
def process_training (data_set):
    train_set = data_set
    train_set = (train_set - np.mean(train_set))/ np.std(train_set)
    train_set= (train_set- np.amin(train_set))/ np.amax(train_set- np.amin(train_set))
    train_set = train_set [:,:,:,np.newaxis]
    
    return train_set

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