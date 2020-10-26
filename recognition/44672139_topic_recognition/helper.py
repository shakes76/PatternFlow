# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob

def map_fn(image_fp, mask_fp):
    image = tf.io.read_file(image_fp)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512, 512))
    image = tf.cast(image, tf.float32) /255.0
    
    mask = tf.io.read_file(mask_fp)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (512, 512))
    mask = mask == [0, 255]
    mask = tf.cast(mask, tf.uint8)
    return image, mask

def display(display_list):
    plt.figure(figsize = (10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
        plt.show()
    
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + tf.keras.backend.epsilon()) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
                                                             + tf.keras.backend.epsilon())

def unet_model(output_channels, f = 64):
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    
    d1 = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(inputs)
    d1 = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(d1)
    
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
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding ='same', activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding ='same', activation='relu')(u3)

    u2 = tf.keras.layers.UpSampling2D()(u3) 
    u2 = tf.keras.layers.concatenate([u2, d2])
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding ='same', activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding ='same', activation='relu')(u2)

    u1 = tf.keras.layers.UpSampling2D()(u2) 
    u1 = tf.keras.layers.concatenate([u1, d1])
    u1 = tf.keras.layers.Conv2D(f, 3, padding ='same', activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(f, 3, padding ='same', activation='relu')(u1)

    #last layer
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(u1)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def show_predictions(ds, model, num=1):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=-1)
        display([tf.squeeze(image), tf.argmax(mask, axis=-1), pred_mask])
        