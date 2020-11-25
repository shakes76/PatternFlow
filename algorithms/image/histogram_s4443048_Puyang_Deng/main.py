#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:21:38 2019

@author: craigdeng
"""

from histogram import *
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from skimage import data


if __name__ =='__main__':
    noisy_image = img_as_ubyte(data.camera())#import image as ubyte
    hist, hist_centers = histogram(noisy_image)#apply the function

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))#plot the image and histogram

    ax[0].imshow(noisy_image, cmap=plt.cm.gray)
    ax[0].axis('off')#adjust original image plot
    
    ax[1].plot(hist_centers, hist, lw=2)
    ax[1].set_title('Histogram of grey values')#adjust histogram plot
    
    plt.tight_layout()