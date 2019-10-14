#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:54 2019

@author: khadekirti
"""

import os 
from view_as_blocks import view_as_blocks
from downscale_local_mean import downscale_local_mean 
 
os.chdir('/Users/khadekirti/Desktop/Pattern Recognistion/Project')

sess = tf.InteractiveSession()  
a = np.arange(15).reshape(3, 5)
image = tf.convert_to_tensor(a)
factor = (2,3) 

b = downscale_local_mean(image, factor ,cval = 0)


 

 