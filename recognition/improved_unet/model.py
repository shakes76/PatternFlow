# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:13:46 2020

@author: s4563609
"""

import tensorflow as tf
import zipfile
import glob

dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
data_path = tf.keras.utils.get_file(origin = dataset_url, fname="/content/keras_png_slices_data.zip")

with zipfile.ZipFile(data_path) as zf:
    zf.extractall()
    
train_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_train/*.png"))
train_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_train/*.png"))
val_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_validate/*.png"))
val_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_validate/*.png"))
test_images = sorted(glob.glob("keras_png_slices_data/keras_png_slices_test/*.png"))
test_masks = sorted(glob.glob("keras_png_slices_data/keras_png_slices_seg_test/*.png"))

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

train_ds = train_ds.shuffle(len(train_images))
val_ds = val_ds.shuffle(len(val_images))
test_ds = test_ds.shuffle(len(test_images))

def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png, channels = 1)
    png = tf.image.resize(png, (256,256))
    return png

def process_path(image_fp, mask_fp):
    image = decode_png(image_fp)
    image = tf.cast(image, tf.float32)/255
    mask = decode_png(mask_fp)
    mask = mask == [0,85,170,255]
    return image, mask

train_ds = train_ds.map(process_path)
val_ds = val_ds.map(process_path)
test_ds = test_ds.map(process_path)

import matplotlib.pyplot as plt

def display(display_list):
    plt.figure(figsize = (10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()    
    
for image, mask in train_ds.take(1):
    display([tf.squeeze(image), tf.argmax(mask, axis=-1)])
    
def unet_model(output_channels, f=64):
    inputs = tf.keras.layers.Input(shape=(256,256,1))
    
    d1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(inputs)
    d1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(d1)
    
    d2 = tf.keras.layers.MaxPooling2D()(d1)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    
    d3 = tf.keras.layers.MaxPooling2D()(d2)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    
    d4 = tf.keras.layers.MaxPooling2D()(d3)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    
    d5 = tf.keras.layers.MaxPooling2D()(d4)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    
    u4 = tf.keras.layers.UpSampling2D()(d5)
    u4 = tf.keras.layers.concatenate([u4, d4])
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    
    u3 = tf.keras.layers.UpSampling2D()(u4)
    u3 = tf.keras.layers.concatenate([u3, d3])
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    
    u2 = tf.keras.layers.UpSampling2D()(u3)
    u2 = tf.keras.layers.concatenate([u2, d2])
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    
    u1 = tf.keras.layers.UpSampling2D()(u2)
    u1 = tf.keras.layers.concatenate([u1, d1])
    u1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(u1)
    
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(u1)
    return tf.keras.Model(inputs=inputs, outputs = outputs)

model = unet_model(4, f=4)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

def show_predictions(ds, num=1):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=-1)
        display([tf.squeeze(image), tf.argmax(mask, axis=-1), pred_mask])
        
show_predictions(val_ds)

from IPython.display import clear_output

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(val_ds)
        
history = model.fit(train_ds.batch(32), epochs=3, 
                    validation_data=val_ds.batch(32), callbacks=[DisplayCallback()])


show_predictions(test_ds, 3)
