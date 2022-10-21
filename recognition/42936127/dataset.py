import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds

seed = 123
batch_size = 64
img_height = 256
img_width = 256

dataset = tf.keras.utils.image_dataset_from_directory(
    "keras_png_slices_data",

    labels = None,
    seed = seed,
    image_size= (img_height, img_width)
    )


# def load_data():

#     data_train = []
#     for filename in os.listdir("keras_png_slices_data/slices/"):
#         image_id:


# def data_generator():


#print("training and validation loaded")

#class_names = dataset.class_names
#print(class_names)

#normalization_layer = tf.keras.layers.Rescaling(1./255)
#normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
#train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
