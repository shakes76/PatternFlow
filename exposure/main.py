#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:52:35 2019

@author: Duc Phan
ID: 44040505
COMP3710
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import skimage.exposure as espo
import exposureTF as etf

from skimage import data, exposure, img_as_float

## Driver scrit
if __name__ == "__main__":
    etf.ex("hihi")
    image = mping.imread('dog1.jpg')
    ##print(image.shape)
    ##imgplot = plt.imshow(image)
    his = espo.histogram(image)
    print(his)
    #plt.show()
    #np.histogram(image, bins=2)
    #exposure.histogram(image, nbins = 2)
