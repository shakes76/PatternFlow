#!/usr/bin/env python
# coding: utf-8

# In[2]:


from os import listdir
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot


def load_brainimg(filename):
    brainimg = Image.open(filename)
    brainimg = brainimg.convert('RGB')
    pixels = asarray(brainimg)
    return pixels

def load_brains(directory, n_brains):
    required_size=(256, 256)
    brains = list()
    for filename in listdir(directory):
        pixels = load_brainimg(directory + filename)
        brainimg = Image.fromarray(pixels)
        brainimg = brainimg.resize(required_size)
        brain = asarray(brainimg)
        if brain is None:
            continue
        brains.append(brain)
        print(len(brains), brain.shape)
        if len(brains) >= n_brains:
            break
    
    return asarray(brains)

directory_train = '../keras_png_slices_data/keras_png_slices_train/'
brains_train = load_brains(directory_train, 5000)
print('Loaded: ', brains_train.shape)
savez_compressed('brain_train.npz', all_brains)


# In[ ]:




