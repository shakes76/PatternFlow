# -*- coding: utf-8 -*-
"""
Loads images from the ISIC dataset in preparation for segmentation via UNet.

@author: s4537175
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import random

"""return make_datasets(
                split_imgs(
                    load_imgs(root_folder), 
                              total_prop, val_prop, test_prop))"""

def get_datasets(root_folder, total_prop, val_prop, test_prop):
    imgs, segs = load_imgs(root_folder)
    train_img, train_seg, val_img, val_seg, test_img, test_seg = split_imgs(imgs, segs, total_prop, val_prop, test_prop)
    return make_datasets(1, train_img, train_seg, val_img, val_seg, test_img, test_seg)

def load_imgs(root_folder):
    ### Load and shuffle image filenames ###
    
    input_path = root_folder + os.path.sep + 'ISIC2018_Task1-2_Training_Input_x2' 
    ground_truth_path = root_folder + os.path.sep + 'ISIC2018_Task1_Training_GroundTruth_x2' 
    #output_path = root_folder + os.path.sep + 'UNet_results'
    
    imgs = glob.glob(input_path + os.path.sep + '*.jpg')
    segs = glob.glob(ground_truth_path + os.path.sep + '*.png')
    
    seed = random.random()
    random.seed(seed)
    random.shuffle(imgs)
    random.seed(seed)
    random.shuffle(segs)
    
    return imgs, segs
    
def split_imgs(imgs, segs, total_prop, val_prop, test_prop):
    # Reduce total number of images to total_prop of original 
    new_img_num = round(len(imgs) * total_prop)
    
    imgs = imgs[0 : new_img_num]
    segs = segs[0 : new_img_num]
    
    print('Total image set size: ' + str(len(imgs)))
    
    # Split remaining images into traing, validation and test sets, according to
    # val_prop and test_prop
    
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
    
    return train_img, train_seg, val_img, val_seg, test_img, test_seg

def make_datasets(img_shape,
                  train_img, train_seg,
                  val_img, val_seg, 
                  test_img, test_seg):
    # Make datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_seg))
    val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_seg))
    test_ds = tf.data.Dataset.from_tensor_slices((test_img, test_seg))
    
    # Shuffle datasets
    train_ds = train_ds.shuffle(len(train_img))
    val_ds = val_ds.shuffle(len(val_img))
    
    train_ds = train_ds.map(load_data)
    val_ds = val_ds.map(load_data)
    test_ds = test_ds.map(load_data)
    
    return train_ds, val_ds, test_ds


def load_img(img_shape, img_file):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    #if (img.shape[0] < img.shape[1]) :
       # tf.image.transpose(img)
    img = tf.image.resize(img, img_shape)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    return img

def load_seg(img_shape, seg_file):
    seg = tf.io.read_file(seg_file)
    seg = tf.image.decode_png(seg, channels=1)
    #if (seg.shape[0] < seg.shape[1]) :
       # tf.image.transpose(seg)
    seg = tf.image.resize(seg, img_shape)
    seg = tf.cast(seg, tf.float32)
    seg = tf.math.round(seg / 255.0)
    bin_seg = (seg == [0.0, 1.0])
    return tf.cast(bin_seg, tf.float32)

def load_data(img_file, seg_file):
    img_shape = (256, 256)
    img = load_img(img_shape, img_file)
    seg = load_seg(img_shape, seg_file)
    return img, seg

def view_imgs(ds, n):
    plt.figure(figsize=(8,n*4))
    i = 0
    for img, label in ds.take(n):
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(img)
        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(label[:,:,1], cmap='gray')
        i = i + 1

def view_preds(model, ds, n):
    plt.figure(figsize=(4*4,n*4))
    i = 0
    for img, true_segs in ds.take(n):
        predictions = model.predict(tf.reshape(img, [1, 256, 256, 3]))
        pred_segs = tf.reshape(predictions, [256, 256, 2])
        
        print(pred_segs[:,:,1])
        print(tf.math.round(pred_segs[:,:,1]))
        print(true_segs[:,:,1])
        
        plt.subplot(n, 4, 4*i + 1)
        plt.imshow(img)
        plt.subplot(n, 4, 4*i + 2)
        plt.imshow(true_segs[:,:,1], cmap='gray')
        plt.subplot(n, 4, 4*i + 3)
        plt.imshow(pred_segs[:,:,1], cmap='gray')
        plt.subplot(n, 4, 4*i + 4)
        plt.imshow(tf.math.round(pred_segs[:,:,1]), cmap='gray')
        i = i + 1
  
#%%

'''
def find_min_size(files):
    min_h = 2000
    min_w = 2000
    portrait = 0;
    for i in range(len(files)):
        print(i)
        img = tf.io.read_file(files[i])
        img = tf.image.decode_png(img, channels=3)
        h = img.shape[0]
        w = img.shape[1]
        if w < h :
            portrait += 1
            temp = w
            w = h
            h = temp
        if h < min_h :
            min_h = img.shape[0]
        if w < min_w :
            min_w = img.shape[1]
    print(min_h)
    print(min_w)
    print(portrait)

find_min_size(imgs)
'''