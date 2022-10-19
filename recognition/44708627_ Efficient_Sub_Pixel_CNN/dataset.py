#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
import os
import math
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[2]:


crop_size = 300
upscale_factor = 4
input_size = crop_size // upscale_factor
batch_size = 8


# In[3]:


def setup_dataset(dir):

    train_ds = image_dataset_from_directory(
        dir,
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
    )

    valid_ds = image_dataset_from_directory(
        dir,
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
    )
    return train_ds, valid_ds


# In[4]:


def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

# Use TF Ops to process.
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


# In[7]:


def test_imgs(test_dir):
    dataset = os.path.join(test_dir, "Combine")

    test_img_paths = sorted(
        [
            os.path.join(dataset, fname)
            for fname in os.listdir(dataset)
            if fname.endswith(".jpeg")
        ]
    )
    return test_img_paths


# In[6]:


def dataset_preprocessing(train_ds,valid_ds):
    # Scale from (0, 255) to (0, 1)
    train_ds = train_ds.map(scaling)
    valid_ds = valid_ds.map(scaling)
    train_ds = train_ds.map(
        lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
    )
    train_ds = train_ds.prefetch(buffer_size=32)
    valid_ds = valid_ds.map(
        lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
    )
    valid_ds = valid_ds.prefetch(buffer_size=32)
    return train_ds, valid_ds

