"""
Module with functions that load images from the ISIC dataset in preparation
for segmentation, and functions which view these images.
"""

# Global variable for image size
image_size = 256

### Imports ###
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import random

### Functions ###

### Main function to be called outside module ###

def get_datasets(root_folder, total_prop, val_prop, test_prop):
    """
    Function which extracts image names from root_folder, shuffles them, and 
    splits them into training, validation and test sets. It then extracts the
    corresponding images and returns 3 image/segmentation datasets.
    """
    imgs, segs = load_imgs(root_folder)
    train_img, train_seg, val_img, val_seg, test_img, test_seg = split_imgs(
        imgs, segs, total_prop, val_prop, test_prop)
    return make_datasets(train_img, train_seg,
                         val_img, val_seg, test_img, test_seg)

### Helper functions for get_datasets ###

def load_imgs(root_folder):
    """
    Function which extracts image and segmentation file names from the
    root_folder location, and randomly shuffles them.
    """
    # Extract all image files from relevant paths
    input_path = (root_folder + os.path.sep +
                  'ISIC2018_Task1-2_Training_Input_x2')
    ground_truth_path = (root_folder + os.path.sep +
                         'ISIC2018_Task1_Training_GroundTruth_x2')
    
    imgs = glob.glob(input_path + os.path.sep + '*.jpg')
    segs = glob.glob(ground_truth_path + os.path.sep + '*.png')

    # Shuffle both the images and segmentation file names with the same random
    # seed so pairs are not mixed up
    seed = random.random()
    random.seed(seed)
    random.shuffle(imgs)
    random.seed(seed)
    random.shuffle(segs)

    # Return shuffled lists
    return imgs, segs
    
def split_imgs(imgs, segs, total_prop, val_prop, test_prop):
    """
    Function which splits the loaded image files into training, validation
    and test sets. Can also reduce overall dataset size.
    """
    # Reduce total number of images to total_prop of original 
    new_img_num = round(len(imgs) * total_prop)
    imgs = imgs[0 : new_img_num]
    segs = segs[0 : new_img_num]
    print('Total image set size: ' + str(len(imgs)))
    
    # Split remaining images into traing, validation and test sets, according 
    # to val_prop and test_prop
    train_prop = 1 - val_prop - test_prop
    train_num = round(len(imgs) * train_prop)
    train_img = imgs[0 : train_num]
    train_seg = segs[0 : train_num]
    
    val_num = round(len(imgs) * val_prop)
    val_img = imgs[train_num : train_num + val_num]
    val_seg = segs[train_num : train_num + val_num]
    
    test_img = imgs[train_num + val_num : len(imgs)]
    test_seg = segs[train_num + val_num : len(imgs)]
    
    print('Training set size: ' + str(len(train_img)))
    print('Validation set size: ' + str(len(val_img)))
    print('Test set size: ' + str(len(test_img)))
    
    # Return the 6 lists of training, validation and testing, image and
    # segmentation file names
    return train_img, train_seg, val_img, val_seg, test_img, test_seg

def make_datasets(train_img, train_seg, val_img, val_seg, test_img, test_seg):
    """
    Function which extracts the corresponding images from files and converts
    6 lists into 3 datasets which are returned. Training and validation datasets
    are set to shuffle.
    """
    # Make datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_seg))
    val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_seg))
    test_ds = tf.data.Dataset.from_tensor_slices((test_img, test_seg))
    
    # Shuffle datasets
    train_ds = train_ds.shuffle(round(len(train_img)/4))
    val_ds = val_ds.shuffle(round(len(val_img)/4))

    # Map filenames into actual images
    train_ds = train_ds.map(load_data)
    val_ds = val_ds.map(load_data)
    test_ds = test_ds.map(load_data)
    
    return train_ds, val_ds, test_ds

### Helper functions for make_datasets, which map image filenames to images ###

def load_img(img_shape, img_file):
    """
    Extracts the corresponding image given a filename.
    Resizes image to image_shape.
    """
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_shape)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    return img

def load_seg(img_shape, seg_file):
    """
    Extracts the corresponding segmentation image given a filename.
    Resizes image to image_shape.
    """
    seg = tf.io.read_file(seg_file)
    seg = tf.image.decode_png(seg, channels=1)
    seg = tf.image.resize(seg, img_shape)
    seg = tf.cast(seg, tf.float32)
    seg = tf.math.round(seg / 255.0)
    bin_seg = (seg == [0.0, 1.0])
    return tf.cast(bin_seg, tf.float32)

def load_data(img_file, seg_file, img_shape=(image_size,image_size)):
    """
    Extracts the corresponding image and segmentation image given both
    filenames.
    Resizes images to image_shape, which is automatically a square of
    image_size (global variable).
    """
    img = load_img(img_shape, img_file)
    seg = load_seg(img_shape, seg_file)
    return img, seg

### Functions to view images ###

def view_imgs(ds, n):
    """
    Creates plot of n images and segmentations from the dataset ds.
    """
    plt.figure(figsize=(8,n*4))
    i = 0
    for img, label in ds.take(n):
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(img)
        plt.subplot(n, 2, 2*i + 2)
        # Plot foreground segmentation
        plt.imshow(label[:,:,1], cmap='gray')
        i = i + 1

def view_preds(model, ds, n):
    """
    Creates plot of n images, true segmentations, and predicted segmentations
    (unrounded and rounded) from the dataset ds, predicted using model.
    """
    plt.figure(figsize=(4*4,n*4))
    i = 0
    for img, true_segs in ds.take(n):
        # Predict segmentations. Must rezise to include and exclude batch size
        predictions = model.predict(
                        tf.reshape(img, [1, image_size, image_size, 3]))
        pred_segs = tf.reshape(predictions, [image_size, image_size, 2])

        plt.subplot(n, 4, 4*i + 1)
        plt.imshow(img)
        # Plot foreground segmentation
        plt.subplot(n, 4, 4*i + 2)
        plt.imshow(true_segs[:,:,1], cmap='gray')
        plt.subplot(n, 4, 4*i + 3)
        plt.imshow(pred_segs[:,:,1], cmap='gray')
        plt.subplot(n, 4, 4*i + 4)
        plt.imshow(tf.math.round(pred_segs[:,:,1]), cmap='gray')
        i = i + 1
