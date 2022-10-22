import os

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

import random as rd

# Used to process images into usable data for model
def process_image(dir):
    AD_train_images = os.listdir(dir + "/train/AD")
    NC_train_images = os.listdir(dir + "/train/NC")
    
    AD_test_images = os.listdir(dir + "/test/AD")
    NC_test_images = os.listdir(dir + "/test/NC")

    #image_size = (240, 256, 3)

    #Train Data
    train_both = []

    for image_name in AD_train_images[:400]:
        image = load_img(dir + "/train/AD/" + image_name, target_size = (128, 128, 3))
        image = img_to_array(image)
        train_both.append([image,1])

    for image_name in NC_train_images[:400]:
        image = load_img(dir + "/train/NC/" + image_name, target_size = (128, 128, 3))
        image = img_to_array(image)
        train_both.append([image,0])

    rd.shuffle(train_both)
    
    train_images = []
    train_labels = []

    for x in train_both:
        train_images.append(x[0])
        train_labels.append(x[1])

    x_train = tf.convert_to_tensor(np.array(train_images, dtype=np.uint8))
    x_train = tf.cast(x_train, tf.float16) / 255.0
    y_train = tf.convert_to_tensor(train_labels)
    
    # Test Data
    test_both = []

    for image_name in AD_test_images[:100]:
        image = load_img(dir + "/test/AD/" + image_name, target_size = (128, 128, 3))
        image = img_to_array(image)
        test_both.append([image,1])

    for image_name in NC_test_images[:100]:
        image = load_img(dir + "/test/NC/" + image_name, target_size = (128, 128, 3))
        image = img_to_array(image)
        test_both.append([image,0])

    rd.shuffle(test_both)
    
    test_images = []
    test_labels = []

    for x in test_both:
        test_images.append(x[0])
        test_labels.append(x[1])

    x_test = tf.convert_to_tensor(np.array(test_images, dtype=np.uint8))
    x_test = tf.cast(x_test, tf.float16) / 255.0
    y_test = tf.convert_to_tensor(test_labels)
    
    
    return x_train, y_train, x_test, y_test