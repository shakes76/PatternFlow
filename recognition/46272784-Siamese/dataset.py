# This file contains the data loader
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tbds
from tensorflow import image
from tensorflow.keras import utils
import matplotlib.pyplot as plt

def transform_images(img):
    # transform to [0,1]
    img = image.rgb_to_grayscale(img)
    img = img / 255
    return img

def loadFile(dir):
    print('--Begin data loading')
    train_AD_dir = os.path.join(dir, 'train/AD')
    train_NC_dir = os.path.join(dir, 'train/NC')
    test_AD_dir = os.path.join(dir, 'test/AD')
    test_NC_dir = os.path.join(dir, 'test/NC')
    print('-Directory of the Training files of class AD is: {}'.format(train_AD_dir))
    print('-Directory of the Training files of class NC is: {}'.format(train_NC_dir))
    print('-Directory of the Testing files of class AD is: {}'.format(test_AD_dir))
    print('-Directory of the Testing files of class NC is: {}'.format(test_NC_dir))
    print('-Loading Training Data of class AD...')
    train_AD_ds = utils.image_dataset_from_directory(train_AD_dir, 
                                                     labels=None,
                                                     label_mode=None,
                                                     image_size=(256, 240),
                                                     shuffle=True,
                                                     batch_size=8)
    print('-Loading Training Data of class NC...')
    train_NC_ds = utils.image_dataset_from_directory(train_NC_dir, 
                                                     labels=None,
                                                     label_mode=None,
                                                     image_size=(256, 240),
                                                     shuffle=True,
                                                     batch_size=8)
    print('-Loading Testing Data of class AD...')
    test_AD_ds = utils.image_dataset_from_directory(test_AD_dir, 
                                                    labels=None,
                                                    label_mode=None,
                                                    image_size=(256, 240),
                                                    shuffle=True,
                                                    batch_size=8)
    print('-Loading Testing Data of class NC...')
    test_NC_ds = utils.image_dataset_from_directory(test_NC_dir, 
                                                    labels=None,
                                                    label_mode=None,
                                                    image_size=(256, 240),
                                                    shuffle=True,
                                                    batch_size=8)
    print('-Mapping datasets to greyscale...')
    train_AD_ds = train_AD_ds.map(transform_images)
    test_AD_ds = test_AD_ds.map(transform_images)
    train_NC_ds = train_NC_ds.map(transform_images)
    test_NC_ds = test_NC_ds.map(transform_images)
    print('--Data loading complete')
    return train_AD_ds, train_NC_ds, test_AD_ds, test_NC_ds
    
def plotExample(ds):
    print(ds)
    print(len(ds))
    for x in ds:
        # print(len(x))
        plt.axis("off")
        plt.imshow((x.numpy()*255).astype("int32")[0])
        plt.show()
        break

# Code for testing the functions
ta, tn, va, vn = loadFile('F:/AI/COMP3710/data/AD_NC/')
plotExample(ta)

