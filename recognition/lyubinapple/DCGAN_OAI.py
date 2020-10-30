'''
    File name: test.py
    Author: Bin Lyu
    Date created: 10/30/2020
    Date last modified: 
    Python Version: 4.7.4
'''
import tensorflow as tf
from os import listdir
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
import glob

def load_images(path, n_images):
    images = list()
    # enumerate files
    for fn in listdir(path):
        # load the image
        image = Image.open(path + fn)
        image = image.convert('RGB')
        pixels = asarray(image)
        # save image into list
        images.append(pixels)
        # stop once we have enough
        if len(images) >= n_images:
            break
    return asarray(images)

def plot_images(imgs, n):
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(imgs[i])
    pyplot.show()

path = "/Users/annalyu/Desktop/Data_Science/2020.2/COMP3710/Ass/Ass3Data/AKOA_Analysis/"
images = load_images(path, 15000)
print('Loaded:', images.shape)
plot_images(images, 5)