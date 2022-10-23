import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_probability as tfp
from statistics import fmean
from PIL import Image
import tensorflow_datasets as tfds

seed = 123
batch_size = 64
img_height = 256
img_width = 256


def calculate_variance(data):
    np_data = tfds.as_numpy(data)
    variances = []
    for batch in np_data:
        for i in range(0, int(len(batch))):
            variances.append(tf.math.reduce_variance(batch[i]))
    return fmean(variances) 


train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "keras_png_slices_data/slices/", 
    labels = None,
    validation_split = 0.3,
    subset = "both",
    seed = seed,
    image_size = (img_height, img_width)
)

train_variance = calculate_variance(train_ds)
