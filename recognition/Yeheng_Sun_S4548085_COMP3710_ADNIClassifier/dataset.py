# import packages
from random import seed
import numpy as np

from tensorflow import keras
import warnings

# ignore warning
warnings.filterwarnings('ignore')

train_data_dir = './ADNI_AD_NC_2D/AD_NC/train'  # train data directory
test_data_dir = './ADNI_AD_NC_2D/AD_NC/test'  # test data directory
class_name_list = ['AD', 'NC']  # list of class name
img_size = 256

# train and validation data loader
def createTrainData(img_size, batch_size):
    train_ds = keras.utils.image_dataset_from_directory(
        directory=train_data_dir,  # target data directory
        labels='inferred',  # data is tagged according to its directory
        label_mode='binary',  # only 2 classes, tagged with value 0 or 1
        batch_size=batch_size,
        image_size=(img_size, img_size),  # the size after resize
        subset = 'validation', ## create validation set
        validation_split = 0.3, ## 30% of train data into validation
        seed = 77
    )
    return train_ds

# test data loader
def createTestData(img_size, batch_size):
    test_ds = keras.utils.image_dataset_from_directory(
        directory=test_data_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=(img_size, img_size),
    )
    return test_ds
