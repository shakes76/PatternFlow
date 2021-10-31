"""
Author: Richard Wainwright
Student ID: 40126812
Date: 05/10/2021

U-net3D model and additional functions for training and assessing performance.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, concatenate, Dropout, MaxPooling3D, \
    BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import SimpleITK as sitk
import matplotlib.pyplot as plt

"""
Labels:
Background = 0
Body = 1
Bones = 2
Bladder = 3
Rectum = 4
Prostate = 5
"""

# Shape of the MRIs of the Prostate 3D data set
IMG_WIDTH = 128
IMG_HEIGHT = 256
IMG_DEPTH = 256
IMG_CHANNELS = 1


def get_image(file_name):
    """
    Read nifti image from file name, create an array and normalise
    :param file_name:
    :return: Normalised array representation of MRI
    """
    sitk_img = sitk.ReadImage(file_name, sitk.sitkFloat32)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    # Normalise by subtracting mean and dividing by standard deviation
    avg = tf.reduce_mean(img_arr)
    sd = tf.math.reduce_std(img_arr)
    img_arr = (img_arr - avg) / sd
    return img_arr


def get_mask(file_name):
    """
    Read image mask from file name and create a one-hot encoding
    :param file_name:
    :return: One-hot array encoding of image mask
    """
    mask = sitk.ReadImage(file_name, sitk.sitkFloat32)
    mask = sitk.GetArrayFromImage(mask)
    # One-hot encoding
    encoding = tf.keras.utils.to_categorical(mask, num_classes=6)
    return encoding


def reshape(dimension, image):
    """
    Reshapes the image to include the dimension required to fit the model
    :param dimension: 1 for MRIs, 6 for masks
    :param image:
    :return: Reshaped array
    """
    return tf.reshape(image, (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, dimension))


def dice(y_test, y_predict, smooth=0.1):
    """
    Calculate the Dice coefficient
    :param y_test: The masks from the test dataset
    :param y_predict: The model predictions
    :param smooth: Smooth factor
    :return: Dice coefficient
    """
    y_test_f = K.flatten(y_test)
    y_pred_f = K.flatten(y_predict)
    intersect = K.sum(y_test_f * y_pred_f)
    d = (2. * intersect + smooth) / (K.sum(y_test_f) + K.sum(y_pred_f) + smooth)
    return d.numpy()


def unet(filters):
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, IMG_CHANNELS))

    # Contraction
    c1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansion
    u6 = Conv3DTranspose(filters * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4], axis=-1)
    c6 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(filters * 16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv3DTranspose(filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3], axis=-1)
    c7 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(filters * 8, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2], axis=-1)
    c8 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(filters * 4, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv3DTranspose(filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=-1)
    c9 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(filters * 2, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv3D(6, (1, 1, 1), activation='softmax')(c9)

    return Model(inputs=[inputs], outputs=[outputs])


def plt_compare(img, test_mask, pred, num):
    """
    Plots the test image, the mask and the model prediction side by side
    :param img: Left-most element
    :param test_mask: Centre element
    :param pred: Right-most element
    :param num: Appended to filename when saved
    """
    # reshape
    img = tf.reshape(img, (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
    test_mask = tf.math.argmax(test_mask, axis=-1)
    pred = tf.math.argmax(pred, axis=-1)

    # plot
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.imshow(img[img.shape[0] // 2], cmap='gray')
    ax1.title.set_text("Image Slice")
    ax2.imshow(test_mask[test_mask.shape[0] // 2], cmap='gray')
    ax2.title.set_text("Test Mask")
    ax3.imshow(pred[pred.shape[0] // 2], cmap='gray')
    ax3.title.set_text("Prediction")
    fig1.savefig("pred_{}.png".format(num))

