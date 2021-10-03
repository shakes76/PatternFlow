# ##### Setup #####

import tensorflow as tf
from tensorflow.keras import layers, models
import PIL

print("Tensorflow Version:", tf.__version__)

# ##### Macros #####

BATCH_SIZE		= 32
IMG_WIDTH		= 260
IMG_HEIGHT		= 228
SEED			= 123
LEARNING_RATE	= 0.001
WEIGHT_DECAY	= 0.0001
EPOCHS			= 50
DROPOUT_RATE	= 0.2
PATCH_DIM		= 2

# ##### Import Data #####

dataDirectory = '../../../AKOA_Analysis'
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
	validation_split=0.2,
	subset="training",
	seed=SEED,
	label_mode=None,
	image_size=(IMG_WIDTH, IMG_HEIGHT),
	batch_size=BATCH_SIZE,
	color_mode='grayscale'
)
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
	validation_split=0.2,
	subset="validation",
	seed=SEED,
	label_mode=None,
	image_size=(IMG_WIDTH, IMG_HEIGHT),
	batch_size=BATCH_SIZE,
	color_mode='grayscale'
)

# Normalize the data to [0,1]
dataset_train = dataset_train.map(lambda a: a / 255.0)
dataset_validation = dataset_validation.map(lambda a: a / 255.0)

# Show some info on the dataset
print(type(dataset_train))
print(len(dataset_train))
print(type(dataset_validation))
print(len(dataset_validation))


