#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:35:55 2019

@author: kajajuel
"""

from PIL import Image

img = Image.open('gray_kitten.jpg')
plt.imshow(img)