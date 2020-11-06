import tensorflow as tf
from tensorflow.keras.models import load_model
from os import path, mkdir
import matplotlib.pyplot as plt

NOISE_INPUT_DIM = 64

class GeneratorLoader():
    def __init__(self):
        self.generator_model = load_model(path.abspath("./output/generator/"))
    def generate(self):
        output_path = path.abspath("./output")
        print_image_path = "{}/image".format(output_path)
        noise_source = tf.random.normal([1, NOISE_INPUT_DIM])
        image = self.generator_model(noise_source)[0]
        # Reverse normalisation
        image = (image + 1) / 2.0
        plt.imshow(image, cmap="gray")
        # Checking whether directory exists
        if not path.exists(output_path):
            raise Exception("No output directory found")
        if not path.exists(print_image_path):
            mkdir(print_image_path)

        plt.savefig(path.abspath("{}/image.png".format(print_image_path)))
