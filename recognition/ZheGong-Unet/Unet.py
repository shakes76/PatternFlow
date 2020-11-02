import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import glob
import zipfile
import numpy as np
from PIL import Image
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import LeakyReLU

dataset_url="https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download"
data_path = tf.keras.utils.get_file(origin=dataset_url,fname="keras_png_slices_data.zip")
with zipfile.ZipFile(data_path) as zf:
    zf.extractall()

train_images=sorted(glob.glob('keras_png_slices_data/keras_png_slices_train/*.png'))
val_images=sorted(glob.glob('keras_png_slices_data/keras_png_slices_validate/*.png'))
test_images=sorted(glob.glob('keras_png_slices_data/keras_png_slices_test/*.png'))
train_masks=sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_train/*.png'))
val_masks=sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_validate/*.png'))
test_masks=sorted(glob.glob('keras_png_slices_data/keras_png_slices_seg_test/*.png'))

train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images,val_masks))
test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_masks))

train_ds = train_ds.shuffle(len(train_images))
val_ds = val_ds.shuffle(len(val_images))
test_ds = test_ds.shuffle(len(test_images))

def decode_png(file_path):
    png = tf.io.read_file(file_path)
    png = tf.image.decode_png(png,channels=1)
    png = tf.image.resize(png,(256,256))
    return png

def process_path(image_fp,mask_fp):
    image = decode_png(image_fp)
    image = tf.cast(image,tf.float32)/255.0
    
    mask = decode_png(mask_fp)
    mask = mask == [0,85,170,255]
    mask = tf.cast(mask,tf.float32)
    return image,mask

train_ds=train_ds.map(process_path)
test_ds=test_ds.map(process_path)
val_ds=val_ds.map(process_path)

import matplotlib.pyplot as plt
def display(display_list):
    plt.figure(figsize=(10,10))
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.imshow(display_list[i],cmap='gray')
        plt.axis('off')
    plt.show()

from tensorflow import argmax
for image,mask in train_ds.take(1):
    display([tf.squeeze(image),tf.argmax(mask,axis=-1)])

def unet_model(f,channel=4):
  inputs = keras.Input(shape=(256,256,1))
  conv1 = tf.keras.layers.Conv2D(4*f, (3, 3), activation='relu', padding='same')(inputs)
  conv2 = tf.keras.layers.Conv2D(4*f, (3, 3), activation='relu', padding='same')(conv1)
  conv2=tf.keras.layers.Dropout(0.3)(conv2)
  conv2 = tf.keras.layers.Conv2D(4*f, (3, 3), activation='relu', padding='same')(conv2) 
  conv2=conv2+conv1

  conv3 = tf.keras.layers.Conv2D(8*f, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv2)
  conv4 = tf.keras.layers.Conv2D(8*f, (3, 3), activation='relu', padding='same',)(conv3)
  conv4=tf.keras.layers.Dropout(0.3)(conv4)
  conv4 = tf.keras.layers.Conv2D(8*f, (3, 3), activation='relu', padding='same',)(conv4)
  conv4=conv4+conv3

  conv5 = tf.keras.layers.Conv2D(16*f, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv4)
  conv6 = tf.keras.layers.Conv2D(16*f, (3, 3), activation='relu', padding='same')(conv5)
  conv6=tf.keras.layers.Dropout(0.3)(conv6)
  conv6 = tf.keras.layers.Conv2D(16*f, (3, 3), activation='relu', padding='same')(conv6)
  conv6=conv6+conv5

  conv7 = tf.keras.layers.Conv2D(32*f, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv6)
  conv8 = tf.keras.layers.Conv2D(32*f, (3, 3), activation='relu', padding='same')(conv7)
  conv8=tf.keras.layers.Dropout(0.3)(conv8)
  conv8 = tf.keras.layers.Conv2D(32*f, (3, 3), activation='relu', padding='same')(conv8)
  conv8=conv8+conv7

  conv9 = tf.keras.layers.Conv2D(64*f, (3, 3), activation='relu', padding='same', strides=(2, 2))(conv8)
  conv10 = tf.keras.layers.Conv2D(64*f, (3, 3), activation='relu', padding='same')(conv9)
  conv10=tf.keras.layers.Dropout(0.3)(conv10)
  conv10 = tf.keras.layers.Conv2D(64*f, (3, 3), activation='relu', padding='same')(conv10)
  conv10=conv10+conv9

  up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv10)
  up1 = tf.keras.layers.Conv2D(32*f, (2, 2), activation='relu', padding='same')(up1)
  up1 = tf.concat([conv8,up1], axis=3)

  up2 = tf.keras.layers.Conv2D(32*f, (3, 3), activation='relu', padding='same')(up1)
  up2 = tf.keras.layers.Conv2D(32*f, (1, 1), activation='relu', padding='same')(up2)
  up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(up2)
  up2 = tf.keras.layers.Conv2D(16*f, (2, 2), activation='relu', padding='same')(up2)
  up2 = tf.concat([conv6,up2], axis=3)

  up3 = tf.keras.layers.Conv2D(16*f, (3, 3), activation='relu', padding='same')(up2)
  up3_ = tf.keras.layers.Conv2D(16*f, (1, 1), activation='relu', padding='same')(up3)
  up3 = tf.keras.layers.UpSampling2D(size=(2, 2))(up3_)
  up3 = tf.keras.layers.Conv2D(8*f, (2, 2), activation='relu', padding='same')(up3)
  up3 = tf.concat([conv4,up3], axis=3)

  up4 = tf.keras.layers.Conv2D(8*f, (3, 3), activation='relu', padding='same')(up3)
  up4_ = tf.keras.layers.Conv2D(8*f, (1, 1), activation='relu', padding='same')(up4)
  up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(up4_)
  up4 = tf.keras.layers.Conv2D(4*f, (2, 2), activation='relu', padding='same')(up4)
  up4 = tf.concat([conv2,up4], axis=3)

  conv = tf.keras.layers.Conv2D(8*f, (3, 3), activation='relu', padding='same')(up4)
  conv = tf.keras.layers.Conv2D(4, (1, 1), activation='relu', padding='same')(conv)
  conv = tf.keras.layers.LeakyReLU(0.01)(conv)

  up3_ = tf.keras.layers.Conv2D(4, (1, 1), padding='same')(up3_)
  up3_ = tf.keras.layers.LeakyReLU(0.01)(up3_)
  up3_ = tf.keras.layers.UpSampling2D(size=(2, 2))(up3_)
  up4_ = tf.keras.layers.Conv2D(4, (1, 1), padding='same')(up4_)
  up4_ = tf.keras.layers.LeakyReLU(0.01)(up4_)
  up4_ = up3_ + up4_
  up4_ = tf.keras.layers.UpSampling2D(size=(2, 2))(up4_)
  conv = conv + up4_

  output = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(conv)
  model=keras.Model(inputs=inputs,outputs=output)
  return model

model = unet_model(10,channel=4)

# from keras import backend as K
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

model.compile(optimizer = tf.keras.optimizers.Adam(lr=5.0e-4), loss = 'categorical_crossentropy', metrics=dice_coef)

model.fit(train_ds.batch(10),epochs=10,validation_data=val_ds.batch(10))

def show_pridicts(ds,num):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis,...])
        pred_mask = tf.argmax(pred_mask[0],axis=-1)
        display([tf.squeeze(image),tf.argmax(mask,axis=-1),pred_mask])

show_pridicts(test_ds,4)

def prediction(ds):
  pred=[]
  true=[]
  for image, mask in ds:
    pred_mask = model.predict(image[tf.newaxis,...])
    pred.append(pred_mask)
    true.append(mask)
  return pred,true

pred,true=prediction(test_ds)

dice = dice_coefficient(true,pred,smooth=1)

print(float(dice))