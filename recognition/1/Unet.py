import os
import numpy as np
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import random

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


# Set the proper GPU index.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set to avoid module warning.
config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth = True  
session = tf.compat.v1.Session(config=config) 

# Hyperparameters.
epochs = 10
batch_size = 4

# Parameters related with this task.
class_number = 4
input_size=(256, 256, 1)

# The dataset size.
train_size = 9664
validate_size = 1120
test_size = 544
train_steps = int((train_size - 1) / batch_size) + 1
validate_steps = int((validate_size - 1) / batch_size) + 1
test_steps = int((test_size - 1) / batch_size) + 1

#1.Build dataloader
# Data generator config.
data_gen_args = dict(rotation_range=0.2,  # Random rotation.
                     width_shift_range=0.05,  # Random width shift.
                     height_shift_range=0.05,  # Random height shift.
                     shear_range=0.05, # Random shear.
                     zoom_range=0.05,  # Random zoom.
                     horizontal_flip=True,  # Random horizontal flip.
                     fill_mode='nearest')  # Use nearest filling mode.

# Set the data generator.
def dataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=True, num_class=class_number, save_to_dir=None, target_size=(256, 256), seed=1):
    # Build the data generator for image and mask.
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    
    # Adjust the data properties.
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        # Rescale the image to [0,1].
        img = img / 255.0

        # Hard-code to get the one-hot label classes.
        mask = np.round(mask)
        batch, height, width, channel = mask.shape
        new_mask = np.zeros((batch, height, width, 4), dtype=np.float32)
        new_mask[:, :, :, 0:1] = mask == 0.0
        new_mask[:, :, :, 1:2] = mask == 85.0
        new_mask[:, :, :, 2:3] = mask == 170.0
        new_mask[:, :, :, 3:4] = mask == 255.0

        yield img, new_mask

# Setup the dataloader for training, validation and test datasets.
trainGen = dataGenerator(
    batch_size,
    './data/',
    'keras_png_slices_train',
    'keras_png_slices_seg_train',
    data_gen_args,
    save_to_dir=None
)
valGen = dataGenerator(
    batch_size,
    './data/',
    'keras_png_slices_validate',
    'keras_png_slices_seg_validate',
    {},
    save_to_dir=None
)
testGen = dataGenerator(
    batch_size,
    './data/',
    'keras_png_slices_test',
    'keras_png_slices_seg_test',
    {},
    save_to_dir=None

)

#2.Build Model.
# Build the UNET model.

def Unetmodel(output_channel = 4):
  input = tf.keras.layers.Input(shape = (256,256,1))


# Image encoder.
# Stage 1.
  conv1 = Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(input)
  conv2 = Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv1)
  conv2 = tf.keras.layers.Dropout(0.3)(conv2)
  conv2 = Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv2)
  sum1 = conv1 + conv2
# Stage 2.
  conv3 = Conv2D(32, 3,2, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(sum1)
  conv4 = Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv3)
  conv4 = tf.keras.layers.Dropout(0.3)(conv4)
  conv4 = Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv4)
  sum2 = conv3 + conv4
# Stage 3.
  conv5 = Conv2D(64, 3,2, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(sum2)
  conv6 = Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv5)
  conv6 = tf.keras.layers.Dropout(0.3)(conv6)
  conv6 = Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv6)
  sum3 = conv5 + conv6

# Stage 4.
  conv7 = Conv2D(128, 3,2, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(sum3)
  conv8 = Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv7)
  conv8 = tf.keras.layers.Dropout(0.3)(conv8)
  conv8 = Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv8)
  sum4 = conv7 + conv8

# Stage 5.
  conv9 = Conv2D(256, 3,2, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(sum4)
  conv10 = Conv2D(256, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv9)
  conv10 = tf.keras.layers.Dropout(0.3)(conv10)
  conv10 = Conv2D(256, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(conv10)
  sum5 = conv9 + conv10

# Image decoder.
# Stage 6.
  up6 = tf.keras.layers.UpSampling2D(size=(2,2))(sum5)
  up6 = Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up6) 
  up6 = concatenate([sum4, up6], axis=3)

# Stage 7.
  up7 = Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up6)
  up7 = Conv2D(128, 1, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up7)
  up7 = tf.keras.layers.UpSampling2D(size=(2,2))(up7)
  up7 = Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up7)
  up7 = concatenate([sum3, up7], axis=3)

# Stage 8.
  up8 = Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up7)
  UP8 = Conv2D(64, 1, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up8)
  UP88 = Conv2D(16, 1, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(UP8)
  UP88 = tf.keras.layers.UpSampling2D(size=(2,2))(UP88)
  up8 = tf.keras.layers.UpSampling2D(size=(2,2))(UP8)
  up8 = Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up8)
  up8 = concatenate([sum2, up8], axis=3)


# Stage 9.
  up9 = Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up8)
  UP9 = Conv2D(32, 1, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up9)
  UP99 = Conv2D(16, 1, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(UP9)
  up9 = tf.keras.layers.UpSampling2D(size=(2,2))(UP9)
  up9 = Conv2D(16, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up9)
  up9 = concatenate([sum1, up9], axis=3)

#Stage 10
  up10 = Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up9)
  up10 = Conv2D(16, 1, activation=tf.keras.layers.LeakyReLU(alpha = 0.01), padding='same')(up10)
  sum6 = UP88+UP99
  sum6 = tf.keras.layers.UpSampling2D(size=(2,2))(sum6)
  sum7 = up10+sum6
  output = Conv2D(16, 1, activation="softmax")(sum7)
  return tf.keras.Model(inputs = input,outputs = output)

# Build the final keras model.
model = Unetmodel(output_channel = 4)

# Show the model structures.
model.summary()

# Setup the optimizer and loss function
smooth=1
def dice_coef(trainGen,testGen, smooth=1):
  intersection = sum(trainGen* testGen, axis=[1,2,3])
  union = sum(trainGen, axis=[1,2,3]) + sum(testGen, axis=[1,2,3])
  dice = mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

#def dice_coef_loss(trainGen,testGen):
#return -dice_coef(trainGen, testGen)

model.compile(optimizer=Adam(lr=1e-4), loss=-dice_coef_loss, metrics=[dice_coef])