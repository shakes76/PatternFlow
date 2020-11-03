import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import glob
import zipfile
import numpy as np
from PIL import Image

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

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

def display(display_list):
    plt.figure(figsize=(10,10))
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.imshow(display_list[i],cmap='gray')
        plt.axis('off')
    plt.show()

def show_pridicts(ds,num):
    for image, mask in ds.take(num):
        pred_mask = model.predict(image[tf.newaxis,...])
        pred_mask = tf.argmax(pred_mask[0],axis=-1)
        display([tf.squeeze(image),tf.argmax(mask,axis=-1),pred_mask])

def prediction(ds):
    pred=[]
    true=[]
    for image, mask in ds:
        pred_mask = model.predict(image[tf.newaxis,...])
        pred.append(pred_mask)
        true.append(mask)
    return pred,true

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

train_ds=train_ds.map(process_path)
test_ds=test_ds.map(process_path)
val_ds=val_ds.map(process_path)

model = unet_model(10,channel=4)
model.compile(optimizer = Adam(lr=5.0e-4), loss = 'categorical_crossentropy', metrics=dice_coef)
model.fit(train_ds.batch(10),epochs=10,validation_data=val_ds.batch(10))
show_pridicts(test_ds,4)

pred,true=prediction(test_ds)
dice = dice_coefficient(true,pred,smooth=1)
print(float(dice))