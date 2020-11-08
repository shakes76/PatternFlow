import tensorflow as tf
import glob
import pathlib
import random
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


def get_unique_values():
    '''
    Return a sorted list of unique classes in segmented image
    '''
    # load the random segmented image for getting classes
    img = load_img(random.choice(train_masks), color_mode='grayscale')
    # convert to numpy array
    img_array = img_to_array(img)

    # convert to 1D array for conveniently finding uniques (classes)
    result = img_array.flatten()
    return np.unique(result)


def image_generator(img_files, mask_files, batch_size=32):
    count = 0
    classes = get_unique_values()
    while (True):
        img = np.zeros((batch_size, 256, 256, 1)).astype('float') # Grayscale images for training
        mask = np.zeros((batch_size, 256, 256, 4)).astype('float') # mask for validation

        for i in range(count, count+batch_size): #initially from 0 to 32, count = 0. 
            train_img = tf.io.read_file(img_files[i])
            train_img = tf.image.decode_png(train_img, channels=1)
            train_img = tf.image.resize(train_img, (256, 256))
            train_img = tf.cast(train_img, tf.float32) / 255.0

            img[i-count] = train_img #add to array - img[0], img[1], and so on.
                                                   

            train_mask = load_img(mask_files[i], color_mode='grayscale')
            # convert to numpy array
            train_mask = img_to_array(train_mask)
            for index, unique_value in enumerate(classes):
                train_mask[train_mask == unique_value] = index
            train_mask = train_mask.reshape(256, 256)
            train_mask = tf.one_hot(train_mask, classes.size, 1, 0, -1, tf.float32)

            mask[i-count] = train_mask

        count+=batch_size
        if(count+batch_size>=len(img_files)):
            count=0
            # print "randomizing again"
        
        yield img, mask


# Dataset directory in same directory as file. OASIS images is in datasets/OASIS
# dataset_path = pathlib.Path(__file__).parent.absolute() / 'datasets/OASIS'
dataset_path = pathlib.Path("/home/long/projects/COMP3710/Assignment3/PatternFlow/recognition/Segmentation/datasets/OASIS")

# Load data
# Each will contains all image directories
train_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_train/*.png")))
train_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_train/*.png")))

test_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_test/*.png")))
test_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_test/*.png")))

val_images = sorted(glob.glob(str(dataset_path / "keras_png_slices_validate/*.png")))
val_masks = sorted(glob.glob(str(dataset_path / "keras_png_slices_seg_validate/*.png")))

# Create image generator for loading processed image to model
train_img_generator = image_generator(train_images, train_masks)
test_img_generator = image_generator(test_images, test_masks)
val_img_generator = image_generator(val_images, val_masks)

count = 0
for img, mask in test_img_generator:
    count += 1
    if count > 1:
        break