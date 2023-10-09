#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:08:45 2019

@author: khadekirti
"""

import tensorflow as tf
import numpy as np 
from view_as_blocks import view_as_blocks



input_ = np.arange(4*4*6).reshape(4,4,6)  

# First -  normal test, as given in sample in the given sklearn    
test1 = view_as_blocks(input_,(1, 2, 2)) 

# Second - if the block shape is negative 
try: 
    test2  = view_as_blocks(input_,(-1, 2, 2)) 
except (ValueError): 
    print("Passed")

# Three - If the shape is different 
try: 
    test3  = view_as_blocks(input_,(1,1, 2, 2)) 
except (ValueError): 
    print("Passed")     
    
    
# Forth - if the block shape not divisible
try: 
    test4  = view_as_blocks(input_,(10, 2, 2)) 
except (ValueError): 
    print("Passed") 
    

# Fifth - if the block shape not tuple
try: 
    test5  = view_as_blocks(input_,[1, 2, 2]) 
except (TypeError): 
    print("Passed")      
 

# Sixth - Increasing the size of the  block shape not tuple
input_ = np.arange(4*4*6*10*20*30).reshape(4,4,6,10,20,30)  
test6  = view_as_blocks(input_,(1, 2, 2, 5 , 5 , 5))
      

