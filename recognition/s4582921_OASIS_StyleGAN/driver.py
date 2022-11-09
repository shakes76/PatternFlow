"""
driver.py

The driver script containing main for the StyleGan2 implementation.

Requirements:
    - tensorflow-gpu - 2.4.1
    - matplotlib - 3.4.3

Author: Bobby Melhem
Python Version: 3.9.7

"""


import argparse
import os
from glob import glob 
from shutil import rmtree

import tensorflow as tf 

import gan


#List of paths to each image
IMAGE_PATHS = glob('./keras_png_slices_data/keras_png_slices_test/*.png') \
    +  glob('./keras_png_slices_data/keras_png_slices_validate/*.png') \
    +  glob('./keras_png_slices_data/keras_png_slices_train/*.png')


def clear_samples(path):
    """
    Deletes all samples from the specified path.

    Args:
        path : the path where the samples are saved.
    """

    old_samples = glob(path + '/*')
    for sample in old_samples:
        rmtree(sample)

    print("Sample Cleared")


def clear_cache(path):
    """
    Deletes the cache at the specified path.

    Args:
        path : the path where the cache is saved.
    """

    rmtree(path)
    os.mkdir(path)

    print("Cache Cleared")


def main():
    """
    Main function to run with driver script.

    Optional Arguments:
        --clear cache | samples | both : clears the cache, samples or both
        --load weights : starts the model by loading latest checkpoint.
    """

    args = parse_args()
    
    if args.clear == "cache":
        clear_cache(gan.CACHE_PATH)
    elif args.clear == "samples":
        clear_samples(gan.SAMPLE_PATH)
    elif args.clear == "both":
        clear_cache(gan.CACHE_PATH)
        clear_samples(gan.SAMPLE_PATH)

    styleGAN = gan.GAN(gan.IMAGE_SIZE, gan.LEARNING_RATE)
    print(styleGAN.generator.model.summary())
    print(styleGAN.discriminator.model.summary())

    styleGAN.load_data(gan.CACHE_PATH, IMAGE_PATHS)

    if args.load == "weights":
        styleGAN.load_weights(gan.CHECKPOINTS_PATH)

    styleGAN.train(gan.EPOCHS)

        
def parse_args():
    """Setup the arguments for main"""

    parser = argparse.ArgumentParser(description='Train StyleGan2')

    parser.add_argument('--clear', required=False)
    parser.add_argument('--load', required=False)

    args =  parser.parse_args()

    return args 


if __name__ == '__main__':
    """Main function"""

    main()





