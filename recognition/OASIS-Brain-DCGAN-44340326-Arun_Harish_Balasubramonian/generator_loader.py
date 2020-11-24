"""
    File name : generator_loader.py
    Author : Arun Harish Balasubramonian
    Student Number : 44340326
    Description : Generator module that uses the model generated to generate
                  a new image.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from os import path, mkdir
import matplotlib.pyplot as plt

NOISE_INPUT_DIM = 64

"""
    Responsible to load the pre-built model and generate the image. 
    Gets used if the user has the --generate-image option set.
"""
class GeneratorLoader():
    def __init__(self):
        # Attempts to load the model expected to be placed in the output directory
        self.generator_model = load_model(path.abspath("./output/generator/"))

    # A function to generate an image and saving them inside the output directory
    def generate(self):
        # path where the expected output is situated
        output_path = path.abspath("./output")
        # Where to place the generated image
        print_image_path = "{}/image".format(output_path)
        # Generate an image from the loaded generator model
        noise_source = tf.random.normal([1, NOISE_INPUT_DIM])
        image = self.generator_model(noise_source)[0]
        # Reverse normalisation
        image = (image + 1) / 2.0
        plt.imshow(image, cmap="gray")
        # Checking whether directory exists
        if not path.exists(output_path):
            raise Exception("No output directory found")
        # If no image directory is present then create one
        if not path.exists(print_image_path):
            mkdir(print_image_path)
        # Save the generated image onto the output/image/ directory
        plt.savefig(path.abspath("{}/image.png".format(print_image_path)))
