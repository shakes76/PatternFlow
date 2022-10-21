"""
File name: data.py
Author: Thomas Chen
Date created: 11/3/2020
Date last modified: 11/24/2020
Python Version: 3
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from setting import *


""" 
split dataset into training and test set, 
for training set, split into actual training set and validation set
"""
# list all images and labels with same order
train_images = sorted(glob(TRAIN_DATA_PATH + IMG_PATH + '/*.jpg'))
train_labels = sorted(glob(TRAIN_DATA_PATH + SEG_PATH + '/*.png'))

# create a dataframe for data generator
df = pd.DataFrame(columns=['image_path', 'label_path'])
for img, label in zip(train_images, train_labels):
    df = df.append({'image_path': os.path.basename(img), 'label_path': os.path.basename(label)}, ignore_index=True)

# split into training and test set
train_data, test_data = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# split the training set into N_FOLDS
kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
folds = list(kf.split(np.zeros(len(train_data))))

# get a fold of cross valid as actual training set and validation set
train_index, val_index = folds[FOLD]
training_data = train_data.iloc[train_index]
validation_data = train_data.iloc[val_index]

print("training_data:", len(training_data))
print("validation_data:", len(validation_data))
print("test_data:", len(test_data))

"""
create training and valid data generators
"""
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

train_img_generator = image_datagen.flow_from_dataframe(
    training_data,
    directory=TRAIN_DATA_PATH + IMG_PATH,
    x_col="image_path",
    class_mode=None,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1  # ensure the same data augmentation is apply to image and mask
)
train_mask_generator = mask_datagen.flow_from_dataframe(
    training_data,
    directory=TRAIN_DATA_PATH + SEG_PATH,
    x_col="label_path",
    class_mode=None,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1
)

val_img_generator = image_datagen.flow_from_dataframe(
    validation_data,
    directory=TRAIN_DATA_PATH + IMG_PATH,
    x_col="image_path",
    class_mode=None,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1
)
val_mask_generator = mask_datagen.flow_from_dataframe(
    validation_data,
    directory=TRAIN_DATA_PATH + SEG_PATH,
    x_col="label_path",
    class_mode=None,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1
)

"""
create test data generators
"""
image_datagen = ImageDataGenerator(rescale=1. / 255)
mask_datagen = ImageDataGenerator(rescale=1. / 255)

test_img_generator = image_datagen.flow_from_dataframe(
    test_data,
    directory=TRAIN_DATA_PATH + IMG_PATH,
    x_col="image_path",
    class_mode=None,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1
)
test_mask_generator = mask_datagen.flow_from_dataframe(
    test_data,
    directory=TRAIN_DATA_PATH + SEG_PATH,
    x_col="label_path",
    class_mode=None,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1
)


def maskGenerator(img_generator, mask_generator):
    for (img, mask) in zip(img_generator, mask_generator):
        # the mask is not a binary image
        mask = (mask > 0.5).astype(np.float32)
        yield img, mask


# create training, validation and test data generator
train_generator = maskGenerator(train_img_generator, train_mask_generator)
val_generator = maskGenerator(val_img_generator, val_mask_generator)
test_generator = maskGenerator(test_img_generator, test_mask_generator)
