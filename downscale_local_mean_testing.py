#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:54 2019

@author: khadekirti
"""

import os 
os.chdir('/Users/khadekirti/Desktop/PRP/PatternFlow')
 

import numpy as np
import tensorflow as tf 
from downscale_local_mean import downscale_local_mean 
 

sess = tf.InteractiveSession()  
a = np.arange(15).reshape(3, 5)
image = tf.convert_to_tensor(a)


# Try first normal test  
test1 =  downscale_local_mean(image,  (2,3) ,cval = 0)

# Second - if the factor is negative 
try: 
   test2  = downscale_local_mean(image,(-2, 3) ,cval = 0) 
except (ValueError): 
    print("ValueError")      
    

# Three - If the shape is different 
try: 
    test3  = downscale_local_mean(image,(1,1, 2)) 
except (ValueError): 
    print("ValueError")     
    
    

# Forth - if the block shape not divisible
test4  =  downscale_local_mean(image,(10,2)) 

    
    

# Fifth - if the block shape not tuple
try: 
    test5 = downscale_local_mean(image,[2,3]) 
except (TypeError): 
    print("ValueError")      
    

# Sixth - if inpiut not tensorflow 
try: 
    test6  =  downscale_local_mean(image.eval(),[2,3]) 
except (TypeError): 
    print("ValueError")      
          
    



 

 