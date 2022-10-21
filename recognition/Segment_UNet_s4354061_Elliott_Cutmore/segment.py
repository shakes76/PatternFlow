
import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# sess = InteractiveSession(config=config)
# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

import tensorflow.keras.backend as K
from tensorflow.keras import datasets, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Dense, BatchNormalization, add, LeakyReLU
# from tensorflow.compat.v1.keras.utils import get_custom_objects
import tensorflow.keras.losses
from PIL import Image, ImageOps
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import pickle

# GLOBAL Variable
# Initialize the random seed
seed = 3


def get_img_target_paths(img_dir, seg_dir):
    """
    This function retrieves the input and segmented images from img_dir and
    seg_dir, respectively. Input images are in .jpg format and segmented
    images are in .png format
    :param img_dir: (str) Image directory for input images to CNN
    :param seg_dir: (str) Image directory for segmented images to CNN
    :return: 2 (list)'s: The first list being the input images filename's found
    in img_dir and the second list is the segmented image filename's found
    in seg_dir
    """
    input_img_paths = sorted(
        [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(seg_dir, fname)
            for fname in os.listdir(seg_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    return input_img_paths, target_img_paths


def get_img_sizes(input_paths):
    """
    This function stores all the images sizes for the image input_paths
    given to it. Helpful for visualising the occurrence/frequency of images
    sizes to be analysed
    :param input_paths: (list) filename's corresponding to images
    :return: (list) of (tuple) tuples are in the form (width, height)
    """
    sizes = [Image.open(f, 'r').size for f in input_paths]
    return sizes


def inspect_images(sizes, range_w=[510, 520], range_h=[380, 390]):
    """
    This function is used to graph the results of the get_img_sizes function.
    The range given is used to zoom into specific parts of the 3d histogram.
    :param sizes: (list) of (tuple) each tuple of the form (width, height)
    :param range_w: (list) [min_width, max_width] of width axis of 3d plot to
     zoom into.
    :param range_h: (list) [min_height, max_height] of height axis of 3d plot
     to zoom into.
    :return: None
    """
    w = [i[0] for i in sizes]
    h = [i[1] for i in sizes]
    w = np.array(w)
    h = np.array(h)

    c = 0  # record the number of plots
    fig = plt.figure(c, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(w, h, bins=10, range=[range_w, range_h])

    # Construct arrays for the anchor positions of the number of bins
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_title("Most frequently occurring low resolution images")
    ax.set_xlabel('Width (pixels)', fontsize=10)
    ax.set_ylabel('Height (pixels)', fontsize=10)
    ax.set_zlabel('Frequency (images)', fontsize=10)

    bin_size = 20
    c += 1
    plt.figure(c, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(w.reshape(len(sizes)), bins=bin_size)
    plt.title("Histogram width")
    plt.subplot(1, 2, 2)
    plt.hist(h.reshape(len(sizes)), bins=bin_size)
    plt.title("Histogram height")

    plt.show()


def train_val_test_split(val_split, input_img_paths,
                         target_img_paths, test_split=0.05, seed=seed):
    """
    This function is used to split up a single dataset of images into a
    training, validation and testing set to be used for training and
    evaluating a CNN.
    :param val_split: (float) [0 to 1] Percentage of the set (after test set
    removed) of images to be used for validating a model when training
    :param input_img_paths: (list) filename's corresponding to the input
    images of the data set
    :param target_img_paths: (list) filename's corresponding to the target
    images of the data set
    :param test_split: (float) [0 to <1] Percentage of the data set to become
    images for testing/evaluating
    :param seed: (int) The seeding number to random number generated shuffling of
    the data set images
    :return: 6 (list)'s 1=training input image filename's, 2=training target
     image filename's, 3=validation input image filename's, 4=validation target
     image filename's, 5=Testing input image filename's, 6=Testing input image
     filename's
    """
    # Split our img paths into a training and a validation set
    test_samples = int(test_split * len(input_img_paths))
    val_samples = int(val_split * (len(input_img_paths) - test_samples))
    train_samples = len(input_img_paths) - test_samples - val_samples

    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    train_input = input_img_paths[:train_samples]
    train_target = target_img_paths[:train_samples]
    val_input = input_img_paths[train_samples:(train_samples + val_samples)]
    val_target = target_img_paths[train_samples:(train_samples + val_samples)]
    test_input = input_img_paths[(train_samples + val_samples):]
    test_target = target_img_paths[(train_samples + val_samples):]

    return train_input, train_target, val_input, val_target, test_input, test_target


def create_UNet(img_dims, num_classes):
    """
    This function creates a typical UNet which is used to segment color (RGB)
    images into binary black and white output segmentation maps.
    :param img_dims: (tuple) (height, width, channels)
    :param num_classes: (int) number of output segmentation tones (2 for binary)
    :return: (tensorflow.keras.Model) instance of a model
    """

    act = 'relu'
    kern = 'he_uniform'
    pad = 'same'
    inter = 'nearest'
    f = [64, 128, 256, 512, 1024]

    # Input layer - has shape (height, width, channels)
    input_layer = Input(shape=img_dims, name="Input")

    ## Convolutional layers - Feature learning
    # VGG 1:
    conv_1_1 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_1_1")(input_layer)
    conv_1_2 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_1_2")(conv_1_1)
    pool_1 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_1")(conv_1_2)

    # VGG 2:
    conv_2_1 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_2_1")(pool_1)
    conv_2_2 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_2_2")(conv_2_1)
    pool_2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_2")(conv_2_2)

    # VGG 3:
    conv_3_1 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_3_1")(pool_2)
    conv_3_2 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_3_2")(conv_3_1)
    pool_3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_3")(conv_3_2)

    # VGG 4:
    conv_4_1 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_4_1")(pool_3)
    conv_4_2 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="conv_4_2")(conv_4_1)
    pool_4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool_4")(conv_4_2)

    # Bottom VGG:
    bot_1 = Conv2D(f[4], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="bot_1")(pool_4)
    bot_2 = Conv2D(f[4], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="bot_2")(bot_1)

    # CONCAT 4:
    cat_4_1 = UpSampling2D((2, 2), interpolation=inter, name="up_4")(bot_2)
    cat_4_1 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_x")(cat_4_1)
    cat_4_2 = Concatenate(axis=3, name="cat_4")([conv_4_2, cat_4_1])
    cat_4_3 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_1")(cat_4_2)
    cat_4_4 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_4_2")(cat_4_3)

    # CONCAT 3:
    cat_3_1 = UpSampling2D((2, 2), interpolation=inter, name="up_3")(cat_4_4)
    cat_3_1 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_x")(cat_3_1)
    cat_3_2 = Concatenate(axis=3, name="cat_3")([conv_3_2, cat_3_1])
    cat_3_3 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_1")(cat_3_2)
    cat_3_4 = Conv2D(f[2], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_3_2")(cat_3_3)

    # CONCAT 2:
    cat_2_1 = UpSampling2D((2, 2), interpolation=inter, name="up_2")(cat_3_4)
    cat_2_1 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_x")(cat_2_1)
    cat_2_2 = Concatenate(axis=3, name="cat_2")([conv_2_2, cat_2_1])
    cat_2_3 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_1")(cat_2_2)
    cat_2_4 = Conv2D(f[1], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_2_2")(cat_2_3)

    # CONCAT 1:
    cat_1_1 = UpSampling2D((2, 2), interpolation=inter, name="up_1")(cat_2_4)
    cat_1_1 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_x")(cat_1_1)
    cat_1_2 = Concatenate(axis=3, name="cat_1")([conv_1_2, cat_1_1])
    cat_1_3 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_1")(cat_1_2)
    cat_1_4 = Conv2D(f[0], (3, 3), activation=act, kernel_initializer=kern, padding=pad, name="cat_1_2")(cat_1_3)

    # Output layer
    output_layer = Conv2D(num_classes, (1, 1), activation="sigmoid", kernel_initializer=kern, padding=pad,
                          name="Output")(cat_1_4)

    # Create model:
    model = Model(inputs=input_layer, outputs=output_layer, name="Model")

    return model


def pre_activation_residual_block(in_layer, f, act, kern, pad):
    """

        :param in_layer: (tf.keras.layer)
        :return: out_layer: (tf.keras.layer)
        """

    # Feedback
    batch_norm_1 = BatchNormalization(name=("PARB_batch_norm_1_%d" % f))(in_layer)
    relu_1 = LeakyReLU(name=("PARB_relu_1_%d" % f))(batch_norm_1)
    conv_1 = Conv2D(f, (3, 3), activation=act, kernel_initializer=kern, padding=pad, name=("PARB_conv_1_%d" % f))(relu_1)
    batch_norm_2 = BatchNormalization(name=("PARB_batch_norm_2_%d" % f))(conv_1)
    relu_2 = LeakyReLU(name=("PARB_relu_2_%d" % f))(batch_norm_2)
    conv_2 = Conv2D(f, (3, 3), activation=act, kernel_initializer=kern, padding=pad, name=("PARB_conv_2_%d" % f))(relu_2)
    out_layer = add([conv_2, in_layer], name=("PARB_sum_%d" % f))

    return out_layer

def context_module(in_layer, f, act, kern, pad):
    """

    :param in_layer: (tf.keras.layer)
    :return: out_layer: (tf.keras.layer)
    """

    PARB = pre_activation_residual_block(in_layer, f, act, kern, pad)
    drop_layer = Dropout(0.3, name=("context_dropout_%d" % f))(PARB)
    conv_1 = Conv2D(f, (3, 3), activation=act, kernel_initializer=kern, padding=pad, name=("context_conv_1_%d" % f))(PARB)
    conv_2 = Conv2D(f, (3, 3), activation=act, kernel_initializer=kern, padding=pad, name=("context_conv_2_%d" % f))(conv_1)
    out_layer = add([conv_2, drop_layer])

    return out_layer

def create_improved_UNet(img_dims, num_classes):
    """
    This function creates an improved UNet based on the paper:
    F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and
    K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics
    Survival Prediction: Contribution to the BRATS 2017 Challenge,”
    Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1
    which is used to segment color (RGB) images into binary black
    and white output segmentation maps.
    :param img_dims: (tuple) (height, width, channels)
    :param num_classes: (int) number of output segmentation tones (2 for binary)
    :return: (tensorflow.keras.Model) instance of a model
    """

    act = 'relu'
    kern = 'he_uniform'
    pad = 'same'
    inter = 'nearest'
    # f = [64, 128, 256, 512, 1024]
    f = [16, 32, 64, 128, 256]
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.01)

    # Input layer - has shape (height, width, channels)
    input_layer = Input(shape=img_dims, name="Input")

    ## Convolutional layers - Feature learning
    # Level 1:
    conv_1 = Conv2D(f[0], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_1")(input_layer)
    context_1 = context_module(conv_1, f[0], act, kern, pad)
    context_add_1 = add([context_1, conv_1], name="context_add_1")

    # Level_2
    conv_2 = Conv2D(f[1], (3, 3), strides=(2, 2), activation=lrelu, kernel_initializer=kern, padding=pad,
                    name="conv_2")(context_add_1)
    context_2 = context_module(conv_2, f[1], act, kern, pad)
    context_add_2 = add([context_2, conv_2])

    # Level_3:
    conv_3 = Conv2D(f[2], (3, 3), strides=(2, 2), activation=lrelu, kernel_initializer=kern, padding=pad,
                    name="conv_3")(context_add_2)
    context_3 = context_module(conv_3, f[2], act, kern, pad)
    context_add_3 = add([context_3, conv_3])

    # Level_4:
    conv_4 = Conv2D(f[3], (3, 3), strides=(2, 2), activation=lrelu, kernel_initializer=kern, padding=pad,
                    name="conv_4")(context_add_3)
    context_4 = context_module(conv_4, f[3], act, kern, pad)
    context_add_4 = add([context_4, conv_4])

    # Level_5:
    conv_5 = Conv2D(f[4], (3, 3), strides=(2, 2), activation=lrelu, kernel_initializer=kern, padding=pad,
                    name="conv_5")(context_add_4)
    context_5 = context_module(conv_5, f[4], act, kern, pad)
    context_add_5 = add([context_5, conv_5])
    up_5 = UpSampling2D((2, 2), interpolation=inter, name="up_5")(context_add_5)
    conv_5_1 = Conv2D(f[3], (3, 3), activation=act, kernel_initializer=kern, padding=pad,
                    name="conv_5_1")(up_5)

    # CONCAT 4:
    # Localisation
    cat_4 = Concatenate(axis=3, name="cat_4")([conv_5_1, context_add_4])
    conv_4_1 = Conv2D(f[3], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_4_1")(cat_4)
    conv_4_2 = Conv2D(f[3], (1, 1), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_4_2")(conv_4_1)
    # Up sampling module
    up_4 = UpSampling2D((2, 2), interpolation=inter, name="up_4")(conv_4_2)
    conv_4_3 = Conv2D(f[2], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad,
                    name="conv_4_3")(up_4)

    # CONCAT 3:
    # Localisation
    cat_3 = Concatenate(axis=3, name="cat_3")([conv_4_3, context_add_3])
    conv_3_1 = Conv2D(f[2], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_3_1")(cat_3)
    conv_3_2 = Conv2D(f[2], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_3_2")(conv_3_1)
    # Segmentation layer:
    seg_3_1 = Conv2D(1, (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="seg_3_1")(conv_3_2)
    seg_3 = UpSampling2D((4, 4), interpolation=inter, name="up_seg_3_2")(seg_3_1)
    # Up sampling module
    up_3 = UpSampling2D((2, 2), interpolation=inter, name="up_3")(conv_3_2)
    conv_3_3 = Conv2D(f[1], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad,
                      name="conv_3_3")(up_3)

    # CONCAT 2:
    # Localisation
    cat_2 = Concatenate(axis=3, name="cat_2")([conv_3_3, context_add_2])
    conv_2_1 = Conv2D(f[1], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_2_1")(cat_2)
    conv_2_2 = Conv2D(f[1], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_2_2")(conv_2_1)
    # Segmentation layer:
    seg_2_1 = Conv2D(1, (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="seg_2_1")(conv_2_2)
    seg_2 = UpSampling2D((2, 2), interpolation=inter, name="up_seg_2")(seg_2_1)
    # Up sampling module
    up_2 = UpSampling2D((2, 2), interpolation=inter, name="up_2")(conv_2_2)
    conv_2_3 = Conv2D(f[0], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad,
                      name="conv_2_3")(up_2)

    # CONCAT 1:
    cat_1 = Concatenate(axis=3, name="cat_1")([conv_2_3, context_add_1])
    conv_1_1 = Conv2D(f[1], (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="conv_1_1")(cat_1)
    seg_1 = Conv2D(1, (3, 3), activation=lrelu, kernel_initializer=kern, padding=pad, name="seg_1")(conv_1_1)
    # Sum seg layers
    seg_sum_2 = add([seg_3, seg_2])
    seg_sum_1 = add([seg_sum_2, seg_1])

    # Output layer
    output_layer = Conv2D(num_classes, (1, 1), activation="sigmoid", kernel_initializer=kern, padding=pad,
                          name="Output")(seg_sum_1)

    # Create model:
    model = Model(inputs=input_layer, outputs=output_layer, name="Model")

    return model


def load_input_image(path, img_dims):
    """
    This function is used to load a single input image of shape/dimensions
    (img_dims) from a given filename (path)
    :param path: (str) filename for image
    :param img_dims: (tuple) (height, width, channels)
    :return: (Tensor) input image data
    """
    img = img_to_array(load_img(path, color_mode='rgb'))
    img = tf.multiply(img, 1 / 255.)
    img = tf.image.resize(img, [img_dims[0], img_dims[1]],
                          preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, img_dims[0], img_dims[1])

    return img


def load_segmented_image(path, img_dims):
    """
    This function is used to load a single target image of shape/dimensions
     (img_dims) from a given filename (path)
    :param path: (str) filename for image
    :param img_dims: (tuple) (height, width, channels)
    :return: (Tensor) shape=[height, width, channel] of target image data
    """
    img = img_to_array(load_img(path, color_mode="grayscale"))
    img = tf.multiply(img, 1 / 255)
    img = tf.image.resize(img, [img_dims[0], img_dims[1]],
                          preserve_aspect_ratio=True)
    img = tf.image.resize_with_crop_or_pad(img, img_dims[0], img_dims[1])

    return img


def load_input_images_from_path_list(batch_input_img_paths, img_dims):
    """
    This function is used to load a batch of input images from storage
    :param batch_input_img_paths: (list) input image filename's
    :param img_dims: (tuple): (height, width, channels)
    :return: (Tensor) of shape=[batch_size, height, width, channel] of input
    image data
    """
    for j, path in enumerate(batch_input_img_paths):
        img = load_input_image(path, img_dims)
        img = tf.reshape(img, [img_dims[0] * img_dims[1] * img_dims[2]])
        img = tf.cast(img, dtype=tf.float32)
        if j == 0:
            x = img
        else:
            x = tf.concat([x, img], axis=0)
    return tf.reshape(x, [len(batch_input_img_paths), img_dims[0],
                          img_dims[1], img_dims[2]])


def load_target_images_from_path_list(batch_target_img_paths, img_dims, num_classes):
    """
    This function is used to load a batch of target images from storage
    :param batch_target_img_paths: (list) input image filename's
    :param img_dims: (tuple) of (height, width, channels)
    :param num_classes: (int) Number of output channels for segmented images
    :return: (Tensor) of shape=[batch_size, height, width, channel] of input
    image data
    """
    for j, path in enumerate(batch_target_img_paths):
        img = load_segmented_image(path, img_dims)
        img = tf.reshape(img, [img_dims[0] * img_dims[1]])
        if num_classes > 1:
            img = tf.keras.utils.to_categorical(img, num_classes)
        img = tf.reshape(img, [img_dims[0] * img_dims[1] * num_classes])
        img = tf.cast(img, tf.uint8)
        if j == 0:
            y = img
        else:
            y = tf.concat([y, img], axis=0)
    return tf.reshape(y, [len(batch_target_img_paths), img_dims[0],
                          img_dims[1], num_classes])


class CustomSequence(Sequence):
    """
    This class is used to generate images for the training and evaluation of
    the UNet CNN model.
    """

    def __init__(self, input_img_paths, target_img_paths, img_dims, batch_size, num_classes):
        """
        CustomSequence init function
        :param input_img_paths: (list) filename's to input images
        :param target_img_paths: (list) filename's to input images
        :param img_dims: (tuple) (height, width, channels)
        :param batch_size: (int) number of images to use for a batch
        :param num_classes: (int) number of segmentation channels
        """
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.num_classes = num_classes

    def __len__(self):
        """
        Returns length of generator
        :return: (int) number of batches for this generator
        """
        return (len(self.target_img_paths) + self.batch_size - 1) // self.batch_size
        # return len(self.target_img_paths) // self.batch_size


    def __getitem__(self, idx):
        """
        Returns a batch of images from this generator
        :param idx: (int) index of batch
        :return: 2 (Tensor)'s: x=batch of input images, y=batch of target
         images
        """

        i = idx * self.batch_size
        end_index = (i + self.batch_size)
        if end_index > len(self.target_img_paths):
            end_index = len(self.target_img_paths)
        batch_input_img_paths = self.input_img_paths[i: end_index]
        batch_target_img_paths = self.target_img_paths[i: end_index]

        x = load_input_images_from_path_list(batch_input_img_paths, self.img_dims)
        y = load_target_images_from_path_list(batch_target_img_paths,
                                              self.img_dims, self.num_classes)

        return x, y

    def on_epoch_end(self):
        """
        Overrides the on_epoch_end method
        :return:
        """
        random.Random(seed).shuffle(self.input_img_paths)
        random.Random(seed).shuffle(self.input_target_paths)


def create_generator(img_paths, target_paths, img_dims, batch_size, num_classes):
    """
    This function is used to create a generator instance of a CustomSequence
    to load batches of images into a model for training of evaluating.
    :param img_paths: (list) input image paths
    :param target_paths: (list) target image paths
    :param img_dims: (tuple) of (height, width, channels) for input images
    :param batch_size: (int) number of images to receive
    :param num_classes: (int) number of output segmentation channels
    :return: (CustomSequence) image generator
    """
    return CustomSequence(img_paths, target_paths, img_dims, batch_size, num_classes)


def dice_coefficient(y_true, y_pred, smooth=0.):
    """
    Function used to evaluate the similarity of two images
    :param y_true: (Tensor) image_1
    :param y_pred: (Tensor) image_2
    :param smooth: (float)
    :return: (float) dice coefficient similarity of two images
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1. - dice_coefficient(y_true, y_pred)


class CustomDSC(tensorflow.keras.losses.Loss):
    def __init__(self, name="CustomDSC"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
            Function used to evaluate the similarity of two images
            :param y_true: (Tensor) image_1
            :param y_pred: (Tensor) image_2
            :param smooth: (float)
            :return: (float) dice coefficient similarity of two images
            """
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - ((2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f)))
        # return dice_coefficient_loss(y_true, y_pred)


def train_model(train_gen, val_gen, model, epochs=1, save_model_path=None,
                save_checkpoint_path=None, save_history_path=None):
    """
    This function is used to train a tensorflow keras model
    :param train_gen: (CustomSequence) generator of training input and
    target images
    :param val_gen: (CustomSequence) generator of validation input and
    target images
    :param model: (tensorflow.keras.Model) instance
    :param epochs: (int) number of epochs to train the model
    :param save_model_path: (str) filename to save trained model to
    :param save_checkpoint_path: (str) filename to save checkpoints to during
    training
    :param save_history_path: (str) filename to save the model's history during
     training to
    :return: (dict) history of model's training
    """

    if save_checkpoint_path:
        print("Saving checkpoints to %s" % save_checkpoint_path)
        checkpoint = ModelCheckpoint(filepath=save_checkpoint_path,
                                     save_weights_only=True,
                                     # save_best_model,
                                     verbose=1)
        callbacks = [checkpoint]
    else:
        callbacks = []

    # Compile the model:
    metrics = ['accuracy']
    loss = dice_coefficient_loss
    model.compile(loss=loss, optimizer=Adam(lr=1e-6), metrics=metrics)

    # Train the model, doing validation at the end of each epoch.
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    history = history.history

    # save the model if a filename was given
    if save_model_path:
        print("Saving model to %s" % save_model_path)
        save_model(model, save_model_path)

    if save_history_path:
        print("Saving history to %s" % save_history_path)
        save_history(history, save_history_path)

    return history


def load_model(load_path):
    """
    Function used to load a pre-trained model
    :param load_path: (str) filename of model to load
    :return: (tensorflow.keras.Model) instance
    """
    if os.path.exists(load_path):

        print("Getting model")
        tensorflow.keras.losses.dice_coefficient_loss = dice_coefficient_loss
        print("Getting custom object")
        model = tf.keras.models.load_model(load_path, compile=False)
        loss = dice_coefficient_loss
        metrics = ['accuracy']
        opt = Adam(lr=1e-6)
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        return model

    else:
        print("File %s not found" % load_path)
        return None


def save_model(model, save_model_path):
    """
    Function to save a tensorflow.keras.Model instance to storage
    :param model: (tensorflow.keras.Model) instance
    :param save_model_path: (str) filename to save model
    :return: None
    """
    model.save(save_model_path)


def save_history(history, history_save_path):
    """
    Function to save history of training a model as a pickle object
    :param history: (dict) history generated by Model.fit(...) in train_model
    :param history_save_path: filename to save history dictionary
    :return: None
    """
    with open(history_save_path, 'wb') as output_file:
        pickle.dump(history, output_file)


def load_history(history_load_path):
    """
    Function to load the history of a pre-trained tensorflow.keras.Model
    from a pickle object
    :param history_load_path: (str)
    :return: (dict) history of model training
    """
    if os.path.exists(history_load_path):
        with open(history_load_path, "rb") as input_file:
            return pickle.load(input_file)
    else:
        print("File %s not found" % history_load_path)
        return None


def check_generator(generator, img_dims, batch_size, num_classes, visualise=False):
    """
    Function used to check if a generator has been created correctly. A
    collection of conditionals and asserts to validate the instance given.
    Also the option to visualise the first batch of images in the generator
    for a sanity check.
    :param generator: (CustomSequence) instance of a generator
    :param img_dims: (tuple) (heigh, width, channel)
    :param batch_size: (int) batch size of images
    :param num_classes: (int) output segmentation channels
    :param visualise: (bool) "True" will show the first batch of images
    :return:
    """
    if not isinstance(generator, CustomSequence):
        print("Generator given is not of type CustomSequence")
        return 0

    if not generator:
        print("Generator is of type None")
        return 0

    # load first batch of images from generator:
    x, y = generator.__getitem__(0)

    if not (isinstance(x, tf.Tensor) and isinstance(x, tf.Tensor)):
        print("items retrieved from generator are not of type Tensor")
        return 0

    shape_x = tf.shape(x)
    if shape_x[0].numpy() < batch_size:
        batch_size = shape_x[0].numpy()
    shape_x = tf.cast(shape_x, dtype=tf.uint8)
    true_shape_x = [batch_size]
    true_shape_x.extend(list(img_dims))
    true_shape_x = tf.convert_to_tensor(true_shape_x)
    true_shape_x = tf.cast(true_shape_x, dtype=tf.uint8)

    shape_y = tf.shape(y)
    shape_y = tf.cast(shape_y, dtype=tf.uint8)
    true_shape_y = [batch_size]
    true_shape_y.extend(list(img_dims)[:-1])
    true_shape_y.append(num_classes)
    true_shape_y = tf.convert_to_tensor(true_shape_y)
    true_shape_y = tf.cast(true_shape_y, dtype=tf.uint8)

    tf.assert_equal(shape_x, true_shape_x)
    tf.assert_equal(shape_y, true_shape_y)
    tf.assert_equal(shape_x[:-1], shape_y[:-1])

    if visualise:

        xn = np.array(x)
        print(np.shape(xn))
        yn = np.array(y)
        # yn = np.argmax(np.array(y), axis=3)
        num_imgs = batch_size
        plt.figure(figsize=(10, 10))
        j = 1
        for i in range(num_imgs):
            plt.subplot(num_imgs, 2, j)
            plt.imshow(xn[i])
            plt.axis("off")
            j = j + 1
            plt.subplot(num_imgs, 2, j)
            plt.imshow(yn[i], cmap='gray')
            plt.axis("off")
            j = j + 1
        plt.tight_layout()
        plt.show()


def training_plot(history):
    """
    Function used to plot the training accuracy and loss of a model when it
     was trained
    :param history: (dict) training history
    :return: None
    """
    length = len(history['val_accuracy'])
    ## ACCURACY
    plt.figure(1, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label = 'val_accuracy')
    plt.plot([0, length], [0.8, 0.8], 'r', linewidth=0.2)
    plt.gca().annotate("80%", xy=(0, 0.80), xytext=(0, 0.80))
    y_max = max(history["val_accuracy"])
    x_max = history["val_accuracy"].index(y_max)
    plt.gca().annotate(str(round(y_max, 5)), xy=(x_max, y_max), xytext=(x_max, y_max))
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.ylim([0.9, 1])
    plt.legend(loc='lower right')

    ## LOSS
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label = 'val_loss')
    y_max = min(history["val_loss"])
    x_max = history["val_loss"].index(y_max)
    plt.gca().annotate(str(round(y_max, 5)), xy=(x_max, y_max), xytext=(x_max, y_max))
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    plt.show()

def evaluate(test_gen, model):
    """
    Function to evaluate a tensorflow.keras.Model with a set of test images
    :param test_gen: (CustomSequence) image generator
    :param model: (tensorflow.keras.Model) instance (trained)
    :return: test_preds=(Tensor) of shape (n, height, width, channels),
    test_loss=(float), test_acc=(float)
    """
    test_loss, test_acc = model.evaluate(test_gen)
    test_preds = model.predict(test_gen)
    return test_preds, test_loss, test_acc


def results(test_input_img_path, test_target_img_path, test_preds, num_imgs, img_dims, num_classes, visualise=False):
    """
    Function used to plot the results of a model evaluation for a given test set.
    In addition calculates the dice scores for all test images.
    :param test_input_img_path: (list) of (str) filename's of test input images
    :param test_target_img_path: (list) of (str) filename's of test target images
    :param test_preds: (Numpy.Array) of output images from
     tensorflow.keras.Model.evaluate()
    :param num_imgs: (int) images to plot
    :param img_dims: (tuple) (height, width, channels)
    :param num_classes: output segmentation channels
    :param visualise: (bool) "True" then input, target and predictions are
     plotted
    :return: None
    """

    max_len = tf.shape(test_preds).numpy()[0]
    if num_imgs > max_len:
        num_imgs = max_len

    print("num_imgs: ", num_imgs)

    test_input_set = test_input_img_path[0:num_imgs]
    test_target_set = test_target_img_path[0:num_imgs]
    x = load_input_images_from_path_list(test_input_set, img_dims)
    y = load_target_images_from_path_list(test_target_set, img_dims, num_classes)
    # dice_targets = np.argmax(np.array(load_target_images_from_path_list(test_target_img_path, img_dims, num_classes)), axis=3)
    dice_targets = np.array(load_target_images_from_path_list(test_target_img_path, img_dims, num_classes))

    xn = np.array(x)
    yn = np.array(y)
    output = np.array(test_preds)
    # yn = np.argmax(np.array(y), axis=3)
    # output = np.argmax(np.array(test_preds), axis=3)

    # for all images in test set
    dice_sim = []
    # print(len(output), len(test_target_img_path))
    for i in range(len(output)):
        dice_sim.append(
            dice_coefficient(tf.convert_to_tensor(dice_targets[i]), tf.convert_to_tensor(output[i])).numpy())

    xn = [a for _, a in sorted(zip(dice_sim, xn), key=lambda pair: pair[0], reverse=True)]
    yn = [a for _, a in sorted(zip(dice_sim, yn), key=lambda pair: pair[0], reverse=True)]
    output = [a for _, a in sorted(zip(dice_sim, output), key=lambda pair: pair[0], reverse=True)]
    dice_sim = sorted(dice_sim);

    print("Dice Coeffiecient Scores: %d images\nAverage Dice Coefficient: %2.4f" % (
    len(dice_sim), tf.math.reduce_mean(dice_sim)))
    for i in range(len(dice_sim)):
        print("Image: %s | Dice: %2.4f" % (test_target_img_path[i].split(os.sep)[-1].split("_")[1], dice_sim[i]))

    if visualise:
        fig, big_axes = plt.subplots(figsize=(10, 3 * num_imgs), nrows=num_imgs, ncols=1)

        for row, big_ax in enumerate(big_axes, start=1):
            big_ax.set_title("Image: %s | Dice: %2.4f" % (
            test_target_img_path[row - 1].split(os.sep)[-1].split("_")[1], dice_sim[row - 1]))
            big_ax.axis("off")
            # # Turn off axis lines and ticks of the big subplot
            # # obs alpha is 0 in RGBA string!
            # big_ax.tick_params(labelcolor=(0, 0, 0, 0.0), top='off', bottom='off', left='off', right='off')
            # # removes the white frame
            # big_ax._frameon = False

        j = 1
        c = 1
        for i in range(num_imgs):
            ax = fig.add_subplot(num_imgs, 3, j)
            ax.imshow(xn[i])  # [:, :, 0], cmap='gray') #
            # ax.tick_params(labelcolor=(0, 0, 0, 0.0), top='off', bottom='off', left='off', right='off')
            ax.set_xlabel('Input')
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            j = j + 1
            ax = fig.add_subplot(num_imgs, 3, j)
            ax.imshow(yn[i], cmap='gray')
            # ax.tick_params(labelcolor=(0, 0, 0 , 0.0), top='off', bottom='off', left='off', right='off')
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            j = j + 1
            ax = fig.add_subplot(num_imgs, 3, j)
            ax.imshow(output[i], cmap='gray')  #
            # ax.tick_params(labelcolor=(0, 0, 0 , 0.0), top='off', bottom='off', left='off', right='off')
            ax.set_xlabel("Predicted")
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            j = j + 1
        plt.tight_layout()

        c += 1
        bin_size = 20
        plt.figure(c, figsize=(10, 5))
        plt.hist(dice_sim, bins=bin_size)
        plt.title("Histogram Dice Similarity Coefficient")
        plt.tight_layout()
        plt.show()



