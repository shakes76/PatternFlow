#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:34:00 2019

@author: khadekirti
"""
import os 
os.chdir('/Users/khadekirti/Desktop/PRP/PatternFlow')

from skimage import data, exposure, img_as_float
from adjust_sigmoid import adjust_sigmoid
# First - Test a normal 
image = img_as_float(data.moon()) 

test1 = adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)
acual_value1  = exposure.adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)

if sum(sum(acual_value1)) == sum(sum(test1)): 
    print( "Test Passed")


test2 = adjust_sigmoid(image, cutoff=0.5, gain=10, inv=True)
acual_value2 = exposure.adjust_sigmoid(image, cutoff=0.5, gain=10, inv=True)


if sum(sum(acual_value2)) == sum(sum(test2)): 
    print( "Test Passed")
 


# Third  - if negative 
image = img_as_float(data.moon()) 
image[0] = -1
try: 
    test3 = adjust_sigmoid(image, 2) 
except ValueError: 
    print("ValueError")   
    

# Forth - if cuttof is negative  
image = img_as_float(data.moon()) 
test4 = adjust_sigmoid(image, -2) 
           

