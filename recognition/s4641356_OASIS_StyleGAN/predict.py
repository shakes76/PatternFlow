import numpy as np
import GANutils
import modules

def generate_images(model_folder: str, num_images = 1, save_folder = None) -> None:
    #TODO docstring
    model = modules.StyleGAN(existing_model_folder= model_folder)
    for i in range(num_images):
        GANutils.create_image()
