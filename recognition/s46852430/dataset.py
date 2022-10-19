# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:57:39 2022

@author: eudre

"""
#importing the libraries
import os 
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

train_path ='C:\\Users\\eudre\\test\\ISIC-2017_Training_Data'
mask_path ='C:\\Users\\eudre\\test\\ISIC-2017_Training_Part1_GroundTruth'

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# Use to delete superpixel image
def delete_super(path):
    os.chdir(train_path)
    for fname in os.listdir(train_path):
        if fname.endswith('superpixels.png') & fname.endswith('.csv'):
            os.remove(fname)

def load_data(train_path, mask_path):
    images = sorted(glob(os.path.join(train_path, "*.jpg")))
    masks = sorted(glob(os.path.join(mask_path, "*.jpg")))
    
    test_size = int(len(images) * 0.2)
    
    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
   
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x                                ## (256, 256, 3)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)                    ## (256, 256)
    x = np.expand_dims(x, axis=-1)              ## (256, 256, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


import tensorflow as tf
from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D, Conv3D
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling3D, Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
tf.random.Generator = None

def conv_block(input_matrix, num_filter, kernel_size, batch_norm):
  X = Conv3D(num_filter,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_matrix)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)

  X = Conv3D(num_filter,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
  
  return X


def modified_UNET(input_img, dropout = 0.2, batch_norm = True):
#Encode
  c1 = conv_block(input_img,8,3,batch_norm)
  p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c1)
  p1 = Dropout(dropout)(p1)
  
  c2 = conv_block(p1,16,3,batch_norm);
  p2 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv_block(p2,32,3,batch_norm);
  p3 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c3)
  p3 = Dropout(dropout)(p3)
  
  c4 = conv_block(p3,64,3,batch_norm);
  p4 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c4)
  p4 = Dropout(dropout)(p4)
  
  c5 = conv_block(p4,128,3,batch_norm);

# Decode
  u6 = Conv3DTranspose(64, (3,3,3), strides=(2, 2, 2), padding='same')(c5);
  u6 = concatenate([u6,c4]);
  c6 = conv_block(u6,64,3,batch_norm)
  c6 = Dropout(dropout)(c6)
  u7 = Conv3DTranspose(32,(3,3,3),strides = (2,2,2) , padding= 'same')(c6);

  u7 = concatenate([u7,c3]);
  c7 = conv_block(u7,32,3,batch_norm)
  c7 = Dropout(dropout)(c7)
  u8 = Conv3DTranspose(16,(3,3,3),strides = (2,2,2) , padding='same')(c7);
  u8 = concatenate([u8,c2]);

  c8 = conv_block(u8,16,3,batch_norm)
  c8 = Dropout(dropout)(c8)
  u9 = Conv3DTranspose(8,(3,3,3),strides = (2,2,2) , padding='same')(c8);

  u9 = concatenate([u9,c1]);

  c9 = conv_block(u9,8,3,batch_norm)
  outputs = Conv3D(4, (1, 1,1), activation='softmax')(c9)
  print("!!!!!!!!!!!!!!!!!!!")
  print(outputs.shape)
  model = Model(inputs=input_img, outputs=outputs)

  return model


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving data """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 4
    lr = 1e-4 ## (0.0001)
    num_epoch = 5


    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(train_path, mask_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)

    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1