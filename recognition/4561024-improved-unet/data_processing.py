'''
Load and process OASIS brain data set

@author Aghnia Prawira (45610240)
'''

import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def test():
    print("Testing data processing.")
    
def load_image(path):
    image_array = []
    for name in os.listdir(path):
        # Loading and resizing image
        filename = path + name
        image = load_img(filename, target_size=(256, 256))
        # Convert image pixels to array
        image = img_to_array(image)
        image = image/255
        image_array.append(image)
    return image_array
        