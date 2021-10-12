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

if __name__ == "__main__":
    print("Tensorflow Version: {tf.__version__}")
    
    img_dir = pathlib.Path("./Data/AKOA_Analysis")
    
    # Get training set
    train_images = tf.keras.utils.image_dataset_from_directory(img_dir,
                                                               labels=None,
                                                               color_mode="grayscale",
                                                               image_size=(260, 228),
                                                               seed=123,
                                                               shuffle=True,
                                                               subset="training",
                                                               validation_split=0.2)
    
    # Get validation set
    valid_images = tf.keras.utils.image_dataset_from_directory(img_dir,
                                                               labels=None,
                                                               color_mode="grayscale",
                                                               image_size=(260, 228),
                                                               seed=123,
                                                               shuffle=True,
                                                               subset="validation",
                                                               validation_split=0.2)
    
    # Normalise the data
    train_images = train_images.map(lambda x: (tf.divide(x, 255)))
    valid_images = valid_images.map(lambda x: (tf.divide(x, 255)))
    
    # Have a look at an image
    for e in train_images:
        plt.imshow(e[0].numpy(), cmap="gray")
        break
    plt.show()
    