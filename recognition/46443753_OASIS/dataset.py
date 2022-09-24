import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images(directories):
    """
    Load the dataset from given directories list.
    Return the result in numpy array format with shape (num_images, 256, 256, 3),
    scaled to 0 and 1. 
    """
    images = []
    for dir in directories:
        for pic_file in tqdm(os.listdir(dir)):
            file_path = os.path.join(dir, pic_file)
            img = load_img(file_path)
            images.append(img_to_array(img) / 255)

    return np.array(images)

# images = load_images(PIC_DIR)
