'''
test infor
'''

import tensorflow as tf
import glob
import matplotlib
import matplotlib.pyplot as plt

from keras.models import *
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.backend as K
print(tf.__version__) 

#Load data
train_images = sorted(glob.glob("D:\keras_png_slices_data\keras_png_slices_train/*.png"))
train_masks = sorted(glob.glob("D:\keras_png_slices_data\keras_png_slices_seg_train/*.png"))
val_images = sorted(glob.glob("D:\keras_png_slices_data\keras_png_slices_validate/*.png"))
val_masks = sorted(glob.glob("D:\keras_png_slices_data\keras_png_slices_seg_validate/*.png"))
test_images = sorted(glob.glob("D:\keras_png_slices_data\keras_png_slices_test/*.png"))
test_masks = sorted(glob.glob("D:\keras_png_slices_data\keras_png_slices_seg_test/*.png"))


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

def display(dgit git isplay_list):
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i], cmap='gray')
        plt.axis('off')
    plt.show()

for image, mask in train_ds.take(1):
    display([tf.squeeze(image), tf.argmax(mask, axis=-1)])

loaded_model = tf.keras.models.load_model('D:\PatternFlow/recognition/test')
model = loaded_model
smooth = 1.
def dice_coef(train_ds, test_ds):
    train_ds_f = K.flatten(train_ds)
    test_ds_f = K.flatten(test_ds)
    intersection = K.sum(train_ds_f * test_ds_f)
    return (2. * intersection + smooth) / (K.sum(train_ds_f) + K.sum(test_ds_f) + smooth)

def dice_coef_loss(train_ds, test_ds):
    return -dice_coef(train_ds, test_ds)
model.compile(optimizer='adam', loss=dice_coef_loss,metrics=[dice_coef])

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

history = model.fit(train_ds.batch(20), epochs=20, validation_data=val_ds.batch(20),callbacks=[DisplayCallback()])
show_predictions(test_ds, 3)