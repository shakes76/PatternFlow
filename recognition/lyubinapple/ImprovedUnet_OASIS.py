'''
    File name: test.py
    Author: Bin Lyu
    Date created: 10/23/2020
    Date last modified: 
    Python Version: 4.7.4
'''
import tensorflow as tf
from os import listdir
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
import glob
import matplotlib.pyplot as plt

train_images = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_train/*.png"))
train_masks = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_seg_train/*.png"))
#print(len(train_masks))
test_images = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_test/*.png"))
test_masks = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_seg_test/*.png"))
val_images = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_validate/*.png"))
val_masks = sorted(glob.glob("/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/keras_png_slices_data/keras_png_slices_seg_validate/*.png"))

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

train_ds = train_ds.shuffle(len(train_images))
val_ds = val_ds.shuffle(len(val_images))
test_ds = test_ds.shuffle(len(test_images))

def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels=1)
    png = tf.image.resize(png, (256, 256))
    return png

def process_path(image_fp, mask_fp):
    image = decode_png(image_fp)
    image = tf.cast(image, tf.float32)/ 255.0

    mask = decode_png(mask_fp)
    mask = mask == [0, 85, 170, 255]
    mask = tf.cast(mask, tf.float32)
    return image, mask

train_ds = train_ds.map(process_path)
val_ds = val_ds.map(process_path)
test_ds = test_ds.map(process_path)

def display(display_list):
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

for image, mask in train_ds.take(1):
    display([tf.squeeze(image), tf.argmax(mask, axis=-1)])
