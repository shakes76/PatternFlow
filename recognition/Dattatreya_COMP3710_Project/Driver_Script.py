# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:39:40 2020

@author: s4558632
"""


import matplotlib.pyplot as plt
plt.style.use("ggplot")




import UNET_Mod_Compile as UNC
import sys

# Providing the input and response image paths from the command line
img_path = str(sys.argv[1])
seg_path = str(sys.argv[2])
img_width = 256
img_height = 256

# Final model creation, compilation, prediction and calculation of Dice Score
UNC.mod_comp(img_path,seg_path,img_height,img_width)

