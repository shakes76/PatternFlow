# %%
import tensorflow as tf 
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
import glob

# %%
def load_images(path):
    image_list = []
    for filename in glob.glob(path+'/*.jpg'):
        img = Image.open(filename).convert('RGB')
        img = img.resize((128, 128))
        img = np.reshape(img, (128, 128, 3)) 
        img = img / 256
        image_list.append(img)

    print('image set shape:', np.array(image_list).shape)
    return np.array(image_list)

# %%
def load_labels(path):
    image_list =[]
    for filename in glob.glob(path+'/*.png'): 
        img = Image.open(filename).convert('RGB')
        img = img.resize((128, 128))
        img = np.reshape(img, (128, 128, 3)) 
        #img = img - 1
        image_list.append(img)

    print('label set shape:', np.array(image_list).shape)
    return np.array(image_list)


# %%
