# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uIEy7pvgJDJNOgLyBOxq-tWtTZ6CxGud
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))

"""## Reference

https://medium.com/the-owl/k-fold-cross-validation-in-keras-3ec4a3a00538
https://github.com/zhixuhao/unet
https://github.com/NifTK/NiftyNet/blob/a383ba342e3e38a7ad7eed7538bfb34960f80c8d/niftynet/layer/loss_segmentation.py
https://gist.github.com/abhinavsagar/fe0c900133cafe93194c069fe655ef6e

## Import
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

"""## Setting"""

IMG_WIDTH = 512
IMG_HEIGHT = 384
IMG_CHANNELS = 1
SEG_IMG_CHANNELS = 2
BATCH_SIZE = 2

TRAIN_DATA_PATH = 'ISIC2018/' 

IMG_PATH = 'ISIC2018_Task1-2_Training_Input_x2'
SEG_PATH = 'ISIC2018_Task1_Training_GroundTruth_x2'

"""## UNet"""

def UNet(input_size=(256, 256, 1), using_sigmoid=False):
    inputs = Input(input_size)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    if using_sigmoid:
        # using sigmoid
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # using softmax
        outputs = Conv2D(2, (1, 1), activation='softmax')(c9)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = UNet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), using_sigmoid=True)
model.summary()

"""## Create dataset"""

train_images = sorted(glob(TRAIN_DATA_PATH + IMG_PATH + '/*.jpg'))
train_labels = sorted(glob(TRAIN_DATA_PATH + SEG_PATH + '/*.png'))

print(len(train_images), len(train_labels))

df = pd.DataFrame(columns=['image_path', 'label_path'])
for img, label in zip(train_images, train_labels):
    df = df.append({'image_path': os.path.basename(img), 'label_path': os.path.basename(label)}, ignore_index=True)

df

kf = KFold(n_splits = 5, random_state = 7, shuffle = True)


train_data, test_data = train_test_split(df, test_size=0.2)
print(len(train_data), len(test_data))

n = len(train_data)
folds = list(kf.split(np.zeros(n)))

train_index, val_index = folds[0]
training_data = train_data.iloc[train_index]
validation_data = train_data.iloc[val_index]
print(len(training_data), len(validation_data), len(test_data))

training_data[:5]

validation_data[:5]

test_data[:5]

"""## Data"""

data_gen_args = dict(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2
)


image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

train_img_generator = image_datagen.flow_from_dataframe(
    training_data, 
    directory = TRAIN_DATA_PATH + IMG_PATH,
    x_col = "image_path", 
    class_mode=None,
    color_mode = 'grayscale',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed=1
)
train_mask_generator = image_datagen.flow_from_dataframe(
    training_data, 
    directory = TRAIN_DATA_PATH + SEG_PATH,
    x_col = "label_path", 
    class_mode=None,
    color_mode = 'grayscale',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed=1
)

val_img_generator = image_datagen.flow_from_dataframe(
    validation_data, 
    directory = TRAIN_DATA_PATH + IMG_PATH,
    x_col = "image_path", 
    class_mode=None,
    color_mode = 'grayscale',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed=1
)
val_mask_generator = image_datagen.flow_from_dataframe(
    validation_data, 
    directory = TRAIN_DATA_PATH + SEG_PATH,
    x_col = "label_path", 
    class_mode=None,
    color_mode = 'grayscale',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed=1
)


image_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator(rescale=1./255)

test_img_generator = image_datagen.flow_from_dataframe(
    test_data, 
    directory = TRAIN_DATA_PATH + IMG_PATH,
    x_col = "image_path", 
    class_mode=None,
    color_mode = 'grayscale',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed=1
)
test_mask_generator = image_datagen.flow_from_dataframe(
    test_data, 
    directory = TRAIN_DATA_PATH + SEG_PATH,
    x_col = "label_path", 
    class_mode=None,
    color_mode = 'grayscale',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed=1
)

def maskGenerator(train_generator):
    for (img,mask) in train_generator:
        mask = (mask > 0.5).astype(np.float32)
        yield (img,mask)

train_generator = maskGenerator(zip(train_img_generator, train_mask_generator))
val_generator = maskGenerator(zip(val_img_generator, val_mask_generator))
test_generator = maskGenerator(zip(test_img_generator, test_mask_generator))

"""## Loss"""

# def dice_coef_fun(smooth=1):
#     def dice_coef(y_true, y_pred):
#         intersection = K.sum(y_true * y_pred, axis=(1,2,3))
#         union = K.sum(y_true, axis=(1,2,3)) + K.sum(y_pred, axis=(1,2,3))
#         sample_dices=(2. * intersection + smooth) / (union + smooth)
#         dices=K.mean(sample_dices,axis=0)
#         return K.mean(dices)
#     return dice_coef
 
# def dice_coef_loss_fun(smooth=0):
#     def dice_coef_loss(y_true,y_pred):
#         return 1-1-dice_coef_fun(smooth=smooth)(y_true=y_true,y_pred=y_pred)
#     return dice_coef_loss

"""## Train"""

model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(train_generator,
          batch_size=BATCH_SIZE,
          steps_per_epoch=400,
          epochs=10,
          callbacks=[model_checkpoint],
          validation_data=val_generator,
          validation_steps=400,
          validation_batch_size=BATCH_SIZE)

model.load_weights("unet.hdf5")
results = model.evaluate(test_generator, verbose=1, steps=len(test_data) / BATCH_SIZE)

results

img, mask = next(test_generator)

print(img.shape, mask.shape)

prediction = model.predict(img)

prediction.shape

batch_id = 0

plt.imshow(prediction[batch_id] >= 0.5)
plt.colorbar()

plt.imshow(img[batch_id])
plt.colorbar()

plt.imshow(mask[batch_id])
plt.colorbar()

