'''
Load and process OASIS brain data set

@author Aghnia Prawira (45610240)
'''

import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def decode_image(filename):
    '''
    Load and resize an image.
    '''
    # Load and resize image
    image = load_img(filename, color_mode='grayscale', target_size=(256, 256))
    # Convert image pixels to array
    image = img_to_array(image, dtype='float32')
    return image
    
def load_image(path):
    '''
    Load and normalize all images in path.
    '''
    image_array = []
    for name in sorted(os.listdir(path))[:200]:
        filename = path + name
        image = decode_image(filename)
        # Normalize image
        image /= 255
        image_array.append(image)
    return image_array

def load_seg(path):
    '''
    Load and one-hot encode all segmentation 
    images in path.
    '''
    seg_array = []
    for name in sorted(os.listdir(path))[:200]:
        filename = path + name
        seg = decode_image(filename)
        # One-hot encode image
        seg = (seg == [0, 85, 170, 255]).astype('float32')
        seg_array.append(seg)
    return seg_array