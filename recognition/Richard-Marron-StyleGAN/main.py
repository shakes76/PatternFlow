"""
    Main driver script for AKOA StyleGAN.
    Image loading and preprocessing happens 
    in this file before being passed into 
    the StyleGAN.
    
    author: Richard Marron
    status: Development
"""


import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import tensorflow as tf
import PIL
import numpy as np
import pathlib
import matplotlib.pyplot as plt

# Directory manipulation libraries
import glob
import natsort as ns

def process_samples(img_paths: list, status: int):
    """
    Find all 2D images pertaining to
    each sample and combine them into a 3D tensor
    
    Params:
        img_paths : List image paths
        status    : 0 = train, 1 = test, 2 = validation
    """
    if status == 0:
        # Train
        path_header = "./Data/keras_png_slices_data/keras_png_slices_train/case_"
    elif status == 1:
        # Test
        path_header = "./Data/keras_png_slices_data/keras_png_slices_test/case_"
    elif status == 2:
        # Validation
        path_header = "./Data/keras_png_slices_data/keras_png_slices_validate/case_"
    else:
        raise ValueError("Incorrect Status Value! Valid Values: 0, 1, 2")
    
    num_start = len(path_header)
    num_stop = num_start + 3
    
    unique_nums = set()
    for f in img_paths:
        # Pull out the sample number
        unique_nums.add(f[num_start:num_stop])
    
    sample_dict = dict()
    for num in iter(unique_nums):
        sample_dict[num] = glob.glob(path_header + num + "*")
        
    for k, v in sample_dict.items():
        print("\n")
        for p in ns.natsorted(v):
            print(p)
    

def main():
    """
    Main Function
    """
    print(f"Tensorflow Version: {tf.__version__}")
    
    train_file_paths = glob.glob("./Data/keras_png_slices_data/keras_png_slices_train/*")
    test_file_paths = glob.glob("./Data/keras_png_slices_data/keras_png_slices_test/*")
    valid_file_paths = glob.glob("./Data/keras_png_slices_data/keras_png_slices_validate/*")
        
    process_samples(train_file_paths, 0)
    
    # test = nib.load(img_dir)
    # print(test)
    
    # Get training set
    # train_images = tf.keras.utils.image_dataset_from_directory(img_dir,
                                                            #    labels=None,
                                                            #    color_mode="grayscale",
                                                            #    image_size=(260, 228),
                                                            #    seed=456,
                                                            #    shuffle=True,
                                                            #    subset="training",
                                                            #    validation_split=0.2)
    # 
    # Get validation set
    # valid_images = tf.keras.utils.image_dataset_from_directory(img_dir,
                                                            #    labels=None,
                                                            #    color_mode="grayscale",
                                                            #    image_size=(260, 228),
                                                            #    seed=456,
                                                            #    shuffle=True,
                                                            #    subset="validation",
                                                            #    validation_split=0.2)
    # 
    # Normalise the data
    # train_images = train_images.map(lambda x: (tf.divide(x, 255)))
    # valid_images = valid_images.map(lambda x: (tf.divide(x, 255)))
    # 
    # Have a look at an image
    # for e in train_images:
        # plt.imshow(e[0].numpy(), cmap="gray")
        # break
    # plt.show()

if __name__ == "__main__":
    # Begin the program
    main()