import tensorflow as tf
import numpy as np
import os
import cv2

import config as c

#helpers
def load_list_from_txt(txt_path, image_dir_path):
    """
        Loads list of images in txt file denoted as:
            `image path relative to image_dir_path` label
    """
    images = []
    labels = []
    print("load_list", txt_path, image_dir_path)
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_dir_path, line[0]))
            labels.append(int(line[1]))

    return images, labels

def load_image_from_path(image_path, label):
    image = []
    try:
        image = cv2.imread(image_path.numpy().decode()).astype(np.float32)
    except Exception as e:
        print("iamge pth", image_path.numpy().decode())
        print(f"coulnd't load {image_path}")
        
        print(e)

    #normalization done elsewhere 
    return image, [label]

#training iterators etc.
def training_data_iterator():
    images, labels = load_list_from_txt(c.train_file_path, c.train_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset.shuffle(len(images))
    # dataset = dataset.repeat()
    #convert image file names to actual image data
    dataset = dataset.map(lambda filename, label: tf.py_function(load_image_from_path, inp=[filename, label], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it 

def test_data_iterator():
    images, labels = load_list_from_txt(c.test_file_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset.shuffle(len(images))
    # dataset = dataset.repeat()
    #convert image file names to actual image data
    dataset = dataset.map(lambda filename, label: tf.py_function(load_image_from_path, inp=[filename, label], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    it = dataset.__iter__()
    return it 
