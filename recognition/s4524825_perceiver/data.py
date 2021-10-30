import tensorflow as tf
import numpy as np
import os
import cv2

import config as c

"""
    data.py provides functions for loading and augmenting images. 
    These were used rather than tensorflow functions to be 
    compatible with older versions of tensorflow that do not have load_from_directory(), 
    and keras preprocessing layers.

"""

def load_list_from_txt(txt_path, image_dir_path):
    """
        Loads list of images in txt file denoted as:
            `image path relative to image_dir_path` label

        The lists applicable are within train.txt and validation.txt, 
        so that the splits are the same as on goliath etc. although could have 
        set seed. 
    """
    images = []
    labels = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_dir_path, line[0]))
            labels.append(int(line[1]))

    arr = []
    for i in range(len(images)):
        arr.append((images[i], labels[i]))

    np.random.shuffle(arr)
    images = [e[0] for e in arr]
    labels = [e[1] for e in arr]

    return images, labels

def random_flip(image):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    return image

#can't use keras layers e.g. keras.layers.Resizing() as not in tf 2.1 for goliath
def normalize(image):
    for i in range(3):
        image[..., i] = (image[..., i] - 127.5) / 127.5
    return image

def load_image_from_path(image_path, label):
    image = []
    try:
        image = cv2.imread(image_path.numpy().decode()).astype(np.float32)
    except Exception as e:
        # print("ima", image_path.numpy().decode())
        print(f"coulnd't load {image_path}")
        print(e)

    # image = image[np.newaxis, :, :, :]
    image = normalize(image)
    # image = random_flip(image)

    return image, [label]

#training iterators etc.
def training_data_iterator():
    images, labels = load_list_from_txt(c.train_file_path, c.train_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # dataset.shuffle(len(images))
    # dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)

    # dataset = dataset.repeat()
    #convert image file names to actual image data
    dataset = dataset.map(lambda filename, label: tf.py_function(load_image_from_path, inp=[filename, label], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it 

def test_data_iterator():
    images, labels = load_list_from_txt(c.test_file_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # dataset.shuffle(len(images))
    # dataset.shuffler()
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.repeat()
    #convert image file names to actual image data
    dataset = dataset.map(lambda filename, label: tf.py_function(load_image_from_path, inp=[filename, label], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it 
