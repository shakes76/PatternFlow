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
from PIL import Image
import cv2
import imageio
import numpy as np
import pathlib
import matplotlib.pyplot as plt

# Directory manipulation libraries
import glob
import natsort as ns

from styleganmodule import StyleGan

def process_samples(img_paths: list, status: int) -> dict:
    """
    Find all 2D images pertaining to
    each sample and combine them into a 3D tensor
    
    Params:
        img_paths : List of image paths
        status    : 0 = train, 1 = test, 2 = validation
    """
    if status == 0:
        # Train
        print("Processing Training Dataset...")
        path_header = "./Data/keras_png_slices_data/keras_png_slices_train/case_"
    elif status == 1:
        # Test
        print("Processing Test Dataset...")
        path_header = "./Data/keras_png_slices_data/keras_png_slices_test/case_"
    elif status == 2:
        # Validation
        print("Processing Validation Dataset...")
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
        # Collect paths for each sample number
        sample_dict[num] = glob.glob(path_header + num + "*")
    
    full_sample = dict()
    for num, dir_list in sample_dict.items():
        # Sort the directories alphanumerically (natural sort)
        dir_list = ns.natsorted(dir_list)
                
        # Collect each slice of a sample into a 3D matrix 
        full_sample[num] = tf.stack(tuple([tf.convert_to_tensor(cv2.imread(dir, cv2.IMREAD_GRAYSCALE), dtype=tf.float32) for dir in dir_list]), axis=-1)
        # Add a channel dimension to make sample 4D
        full_sample[num] = tf.expand_dims(full_sample[num], axis=-1)
        # Normalise the data between -1 and 1
        full_sample[num] = full_sample[num] / 127.5 - 1

    return full_sample

def show_example(name: str, sample):
    """
    Creates a gif of 3D sample
    
    Params:
        name   : The name of the file to create
        sample : The sample from which to create a gif
    """
    print(f"Now creating {name}.gif")
    with imageio.get_writer(f"{name}.gif", mode='I') as writer:
        for i in range(sample.shape[2]):
            writer.append_data(tf.cast((sample[:, :, i, 0] + 1) * 127.5, dtype=tf.uint8).numpy())
    print("Creation complete!")

def main():
    """
    Main Function
    """
    print(f"Tensorflow Version: {tf.__version__}")
    
    train_file_paths = glob.glob("./Data/keras_png_slices_data/keras_png_slices_train/*")
    test_file_paths = glob.glob("./Data/keras_png_slices_data/keras_png_slices_test/*")
    valid_file_paths = glob.glob("./Data/keras_png_slices_data/keras_png_slices_validate/*")
    
    # Process the data
    train_samples = process_samples(train_file_paths, 0)
    test_samples = process_samples(test_file_paths, 1)
    valid_samples = process_samples(valid_file_paths, 2)
    
    # Create a gif of one of the samples
    show_example("example_test", list(train_samples.values())[0])
    

if __name__ == "__main__":
    # Begin the program
    main()