"""
Author: Joshua Knowles
Student ID: 45525803
Date: 27/10/2021

Contains the functions and classes required to build the 3D U-Net model.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, UpSampling3D, concatenate
from tensorflow.keras import backend as K

import nibabel as nib

from pyimgaug3d.utils import to_channels

N_CLASSES = 6

def unet3d_model(input_size=(256,256,128,1), n_classes=6):
    """
    3D U-Net model, implemented as described in https://arxiv.org/abs/1606.06650,
    but with a smaller number of filters at each 3D convolutional layer.
    """

    inputs = Input(input_size)
    
    conv1 = Conv3D(8, (3,3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
    
    pool1 = MaxPool3D((2,2,2))(conv1)
    
    conv2 = Conv3D(16, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv2)
    
    pool2 = MaxPool3D((2,2,2))(conv2)
    
    conv3 = Conv3D(32, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv3)
    
    pool3 = MaxPool3D((2,2,2))(conv3)
    
    conv4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv4)
    
    upconv4 = UpSampling3D(size=2)(conv4)
    upconv4 = Conv3D(64, (2,2,2), activation='relu', padding='same')(upconv4)
    
    concat_3_5 = concatenate([conv3, upconv4], axis=4)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(concat_3_5)
    conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv5)
    
    upconv5 = UpSampling3D(size=2)(conv5)
    upconv5 = Conv3D(32, (2,2,2), activation='relu', padding='same')(upconv5)
    
    concat_2_6 = concatenate([conv2, upconv5], axis=4)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(concat_2_6)
    conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv6)
    
    upconv6 = UpSampling3D(size=2)(conv6)
    upconv6 = Conv3D(16, (2,2,2), activation='relu', padding='same')(upconv6)
    
    concat_1_7 = concatenate([conv1, upconv6], axis=4)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(concat_1_7)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)
    
    output_seg = Conv3D(n_classes, (1,1,1), activation='softmax')(conv7)
    
    unet3d = tf.keras.Model(inputs=inputs, outputs=output_seg)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    unet3d.compile (optimizer=opt, loss='CategoricalCrossentropy', metrics=[average_dice_coefficient])
    
    return unet3d

def dice_coefficient(y_true, y_pred):
    """
    Implementation of the Sørensen–Dice coefficient.
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    total_area = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return 2 * intersection / total_area

def average_dice_coefficient(y_true, y_pred):
    """
    Computes the Sørensen–Dice coefficient for each label dimension and returns
    the average.
    """

    total_dc = 0

    for i in range(N_CLASSES):

        y_true_f = K.flatten(y_true[...,i])
        y_pred_f = K.flatten(y_pred[...,i])
        
        intersection = K.sum(y_true_f * y_pred_f)
        total_area = K.sum(y_true_f) + K.sum(y_pred_f)

        total_dc += 2 * intersection / total_area
    
    return total_dc / N_CLASSES

class MRISequence(tf.keras.utils.Sequence):
    """
    Overriden Tensorflow Sequence to lazy load the processed MRI and label files.
    Uses a batch size of 1 due to high memory usage.
    """

    def __init__(self, x_set, y_set, batch_size=1):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # As the batch size is 1, this method simply returns the indexed mri and label

        mri_filename = self.x[idx]
        label_filename = self.y[idx]
        
        mri = nib.load(mri_filename).get_fdata()[None, ..., None]
        label = to_channels(nib.load(label_filename).get_fdata())[None,...]
        
        return mri, label