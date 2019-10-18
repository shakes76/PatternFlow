#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:34:00 2019

@author: khadekirti
"""

from skimage import data, exposure, img_as_float
# First - Test a normal 
image = img_as_float(data.moon()) 

test1 = adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)

# Second - if negative 
image = img_as_float(data.moon()) 
image[0] = -1
try: 
    im = adjust_gamma(image, 2) 
except ValueError: 
    print("ValueError")   
    
# Second - if image is negative  
image = img_as_float(data.moon()) 
image[0] = -1
try: 
    im = adjust_gamma(image, 2) 
except ValueError: 
    print("ValueError")       
    
    
# Third - if gamma is negative  
image = img_as_float(data.moon()) 
try: 
    im = adjust_gamma(image, -2) 
except ValueError: 
    print("ValueError")            