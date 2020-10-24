import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

PATH_ORIGINAL_DATA = "data/image"
PATH_SEG_DATA = "data/mask"
# IMAGE_HEIGHT = 32
# IMAGE_WIDTH = 32
SEED = 45
BATCH_SIZE = 32


image_data_generator = keras.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

mask_data_generator = keras.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

image_train_gen = image_data_generator.flow_from_directory(
    PATH_ORIGINAL_DATA,
    seed=SEED,
    class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE)

image_test_gen = image_data_generator.flow_from_directory(
    PATH_ORIGINAL_DATA,
    seed=SEED,
    class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE)

mask_train_gen = image_data_generator.flow_from_directory(
    PATH_SEG_DATA,
    seed=SEED,
    class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE)

mask_test_gen = image_data_generator.flow_from_directory(
    PATH_SEG_DATA,
    seed=SEED,
    class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE)