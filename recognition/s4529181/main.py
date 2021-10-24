# author: Yaoyu Liu 

import glob
import tensorflow as tf
from unet import *
import keras.backend as K
import matplotlib.pyplot as plt

# process the images and labels
def process_images_labels(image, label):
    myimage = tf.io.read_file(image)
    myimage = tf.io.decode_jpeg(myimage, channels=3)
    myimage = tf.image.resize(myimage, (256, 256))
    myimage = tf.cast(myimage, tf.float32) / 255.0
    myimage.set_shape([256, 256, 3])

    mylabel = tf.io.read_file(label)
    mylabel = tf.io.decode_png(mylabel, channels=0)
    mylabel = tf.image.resize(mylabel, (256, 256))
    mylabel = tf.squeeze(mylabel)
    mylabel = tf.expand_dims(mylabel, -1)
    mylabel = tf.keras.backend.round(mylabel / 255.0)
    mylabel.set_shape([256, 256, 1])
    return myimage, mylabel

# Displays n images and labels from the ds dataset.
def display_data(ds, n=1):
    for image, label in ds.take(n):
        draws = [tf.squeeze(image), tf.squeeze(label)]
        plt.figure(figsize=(10, 6))
        for i in range(len(draws)):
            plt.subplot(1, len(draws), i+1)
            plt.imshow(draws[i], cmap='gray')
            plt.axis('off')
        plt.show()

# to get data
images = glob.glob(
    'C:\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2')
labels = glob.glob(
    'C:\ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2')

# split the images into train, validate and test datasets
train_end = int(len(images)*0.5)
val_end = int(len(images)*0.7)

train_ds = tf.data.Dataset.from_tensor_slices(
    (images[:train_end], labels[:train_end]))
val_ds = tf.data.Dataset.from_tensor_slices(
    (images[train_end:val_end], labels[train_end:val_end]))
test_ds = tf.data.Dataset.from_tensor_slices(
    (images[val_end:], labels[val_end:]))

# shuffle datasets
train_ds = train_ds.shuffle(train_end)
val_ds = val_ds.shuffle(val_end - train_end)
test_ds = test_ds.shuffle(len(images) - val_end)

# Map datasets to pre-processing function
train_ds = train_ds.map(process_images_labels)
val_ds = val_ds.map(process_images_labels)
test_ds = test_ds.map(process_images_labels)

# plot example image
display_data(train_ds)

# Build model
model = model_unet(16)
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['dice']) #or accuracy