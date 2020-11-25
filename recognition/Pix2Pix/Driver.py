#!/usr/bin/env python
# coding: utf-8


from Pix2Pix_Algorithm import run_Pix2Pix, generate_images
import os
from tqdm import tqdm_notebook, tnrange
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.transform import resize
import tensorflow as tf


## setting Batch size and epoch values. 
batch_size = 24
epoch = 50

# calling run_pix2Pix() function from Pix2Pix_algorithm to strat model training.
run_Pix2Pix(batch_size, epoch)


"""
Loading test dataset 
"""

imgName_X_test = next(os.walk("/keras_png_slices_data/keras_png_slices_seg_test"))[2] # list of names all images in the given path
imgName_y_test = next(os.walk("/keras_png_slices_data/keras_png_slices_test"))[2] # list of names all images in the given path

print("No. of testing images = ", len(imgName_X_test))
print("No. of testing images labels = ", len(imgName_y_test))    

X_test = np.zeros((len(imgName_X_test), 256, 256, 1), dtype=np.float32)
y_test = np.zeros((len(imgName_y_test), 256, 256, 1), dtype=np.float32)



"""
Data Preprocessing on test dataset. 
"""
########################## For Testing #######################################################
for n_test, id_test in tqdm_notebook(enumerate(imgName_X_test), total=len(imgName_X_test)):
    # Loading validating images
    img_test = load_img("/keras_png_slices_data/keras_png_slices_seg_test/"+id_test, grayscale=True)
    x_img_test = img_to_array(img_test)
    x_img_test = resize(x_img_test, (256, 256, 1), mode = 'constant', preserve_range = True)
    
    X_test[n_test] = (x_img_test / 127.5) - 1 #### making pixel values between -1 & 1

for n_mask_test, id_mask_test in tqdm_notebook(enumerate(imgName_y_test), total=len(imgName_y_test)):
    # Loading validating images
    mask_test = load_img("/keras_png_slices_data/keras_png_slices_test/"+id_mask_test, grayscale=True)
    y_img_test = img_to_array(mask_test)
    y_img_test = resize(y_img_test, (256, 256, 1), mode = 'constant', preserve_range = True)
    
    # Save images
    y_test[n_mask_test] = (y_img_test / 127.5) - 1   ### making pixel values between -1 & 1
    
    
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


"""
Checking trained GAN model's performance on test dataset.
"""

for inp, tar in test_ds.take(2):  #Taking 2 images from test dataset.
    #reshaping images to 4d which is a requirement of trained GAN model.
    inp_test = tf.reshape(inp, [1,256,256,1]) 
    tar_test = tf.reshape(tar, [1,256,256,1])
    
    #GAN mopdel's performce is measured
    generate_images(inp_test, tar_test) # generate_images() function present in Pix2Pix_Algorithm generates images using generator model in GAN.
