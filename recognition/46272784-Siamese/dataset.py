# This file contains the data loader
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import utils
import matplotlib.pyplot as plt

def transform_images(image, label):
    # transform to [0,1]
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255
    return image, label

def loadFile(dir):
    train_dit = dir + '/train'
    test_dit = dir + '/test'
    train_ds = utils.image_dataset_from_directory(train_dit, 
                                                  labels='inferred',
                                                  label_mode='int',
                                                  class_names=['AD', 'NC'],
                                                  image_size=(256, 240),
                                                  shuffle=True,
                                                  batch_size=8)
    test_ds = utils.image_dataset_from_directory(test_dit, 
                                                 labels='inferred',
                                                 label_mode='int',
                                                 class_names=['AD', 'NC'],
                                                 image_size=(256, 240),
                                                 shuffle=True,
                                                 batch_size=8)
    train_ds = train_ds.map(transform_images)
    return train_ds, test_ds
    
def plotExample(ds):
    # print(ds)
    for x in ds:
        # print(x[0][0][100])
        # print(x[1])
        # print(len(x))
        plt.axis("off")
        plt.imshow((x[0].numpy()*255).astype("int32")[0])
        plt.show()
        break

# Code for testing the functions
train_ds, test_ds = loadFile(r'F:\AI\COMP3710\data\AD_NC')
plotExample(train_ds)

