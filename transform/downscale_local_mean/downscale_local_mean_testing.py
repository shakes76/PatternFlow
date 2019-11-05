#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:54 2019

@author: khadekirti


For Downscale Local mean, things that needs to check are
- Check if the function is working
- Check if the the function throws a error for negative factor
- Check if the the funtion throws an error if the shape of downscale is different from inout size
- Check if block shape is not divisable
- Check if the block shape is not tuple
- Test with cval
- Test with large image

"""

import numpy as np 
from downscale_local_mean import downscale_local_mean 
 

image = np.arange(15).reshape(3, 5)
downscale_local_mean(image, (2, 3)) 


# First - normal test, as per example  
test1 =  downscale_local_mean(image,  (2,3) ,cval = 0)

# Second - if the factor is negative 
try: 
   test2  = downscale_local_mean(image,(-2, 3) ,cval = 0) 
except (ValueError): 
    print("Passed")      
    

# Three - If the shape is different 
try: 
    test3  = downscale_local_mean(image,(1,1, 2)) 
except (ValueError): 
    print("Passed")     
    
    
# Forth - if the block shape not divisible
test4  =  downscale_local_mean(image,(10,2)) 

    
# Fifth - if the block shape not tuple
try: 
    test5 = downscale_local_mean(image,[2,3]) 
except (TypeError): 
    print("Passed")      
    
 
# Sixth - change the cval
test6 =  downscale_local_mean(image,  (2,3) ,cval = 2)
      

# seveth - large image 
image = np.arange(200).reshape(2, 2,5,10)
test7  =  downscale_local_mean(image,(2,3,2,2)) 

    



 

 
