import tensorflow as tf
import numpy as np

DATA_DIR = './data/keras_png_slices_data'
TRAIN_DATA = DATA_DIR + '/keras_png_slices_train/'
TEST_DATA = DATA_DIR + '/keras_png_slices_test/'
VALIDATE_DATA = DATA_DIR + '/keras_png_slices_validate/'

x_train = tf.keras.utils.image_dataset_from_directory(TRAIN_DATA, labels=None)
x_validate = tf.keras.utils.image_dataset_from_directory(VALIDATE_DATA, labels=None)
x_test = tf.keras.utils.image_dataset_from_directory(TEST_DATA, labels=None)

# get mean, variance of training set
data_info = x_train.reduce((0, 0, 0), 
        lambda x, y: x + (
            1,                                  # n
            y,                                  # running mean
            (x[0]*y - x[1])**2/(x[0]*(x[0]+1))  # running variance
        )
    )

data_mean = data_info[1]/data_info[0]
data_var = data_info[2]/data_info[0]

# Basic scaling and preprocessing
def scaling(data):
    return (data/255.0) - data/255.0

x_train = x_train.apply(scaling)
x_test = x_test.apply(scaling)
x_validate = x_validate.apply(scaling)
