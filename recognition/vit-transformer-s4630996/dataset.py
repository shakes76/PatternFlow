########################################################################################################
######################################    Import data function  ########################################
########################################################################################################

from tensorflow import keras
from config import *


def import_data(IMAGE_SIZE, BATCH_SIZE, paths):
    """ Import data for use in the ViT model.  Returns a training, validation, and test set """

    # extract paths from dictionary argument
    path_train = paths['training']
    path_validate = paths['validation']
    path_test = paths['test']

    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    data_train = keras.preprocessing.image_dataset_from_directory(
        path_train,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123,
        validation_split=None,
        smart_resize=True)

    data_validate = keras.preprocessing.image_dataset_from_directory(
        path_validate,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123,
        validation_split=None,
        smart_resize=True)

    data_test = keras.preprocessing.image_dataset_from_directory(
        path_test,
        labels="inferred",
        label_mode="binary",
        color_mode="grayscale",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123,
        smart_resize=True)
    
    return data_train, data_validate, data_test
