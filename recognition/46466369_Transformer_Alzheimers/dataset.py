import os
from pathlib import Path
import numpy as np
from PIL import Image
import zipfile

"""
dataset.py
Contains the data loader for loading and preprocessing the data
"""

# Path to dataset directory
path = 'C:/Users/Eric/Documents/uni/2022_sem2/comp3710/report/'
#path = '/content/gdrive/MyDrive/ADNI/train_AD.zip'

# Load all the training images
def load_dataset(size):

    zipped_images = zipfile.ZipFile(f'{path}train_AD.zip')
    images_train_AD = np.array([np.array(Image.open(zipped_images.open(image)).resize((size, size))) for image in zipped_images.infolist()])
    zipped_images = zipfile.ZipFile(f'{path}train_NC.zip')
    images_train_NC = np.array([np.array(Image.open(zipped_images.open(image)).resize((size, size))) for image in zipped_images.infolist()])
    zip = zipfile.ZipFile(f'{path}test_AD.zip')
    images_test_AD = np.array([np.array(Image.open(zip.open(image)).resize((size, size))) for image in zip.infolist()])
    zip = zipfile.ZipFile(f'{path}test_NC.zip')
    images_test_NC = np.array([np.array(Image.open(zip.open(image)).resize((size, size))) for image in zip.infolist()])
    
    # encode AD as 1 and NC as 0
    images_test_labels = np.concatenate((np.ones(images_test_AD.shape[0]), np.zeros(images_test_NC.shape[0])))
    images_test = np.concatenate((images_test_AD, images_test_NC), axis = 0)
    images_train_labels = np.concatenate((np.ones(images_train_AD.shape[0]), np.zeros(images_train_NC.shape[0])))
    images_train = np.concatenate((images_train_AD, images_train_NC), axis = 0)
    print("Data successfully loaded")

    # reshape to 4D tensors
    train = np.repeat(images_train[..., np.newaxis], 3, -1)
    test = np.repeat(images_test[..., np.newaxis], 3 -1)

    # reshape to 3D tensors
    trainy = images_train_labels.reshape(images_train_labels.shape[0], 1)
    testy = images_test_labels.reshape(images_test_labels.shape[0], 1)
    return (train, trainy, test, testy)