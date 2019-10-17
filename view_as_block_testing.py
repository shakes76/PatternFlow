#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:08:45 2019

@author: khadekirti
"""

import os 
os.chdir('/Users/khadekirti/Desktop/PRP/PatternFlow')
 
import tensorflow as tf
import numpy as np 
from view_as_blocks import view_as_blocks



A = np.arange(4*4*6).reshape(4,4,6)  

# Try first normal test  
test1 = view_as_blocks(A,(1, 2, 2)) 

# Second - if the block shape is negative 
try: 
    test2  = view_as_blocks(A,(-1, 2, 2)) 
except (ValueError): 
    print("ValueError")

# Three - If the shape is different 
try: 
    test3  = view_as_blocks(A,(1,1, 2, 2)) 
except (ValueError): 
    print("ValueError")     
    
    
# Forth - if the block shape not divisible
try: 
    test4  = view_as_blocks(A,(10, 2, 2)) 
except (ValueError): 
    print("ValueError") 
    

# Fifth - if the block shape not tuple
try: 
    test4  = view_as_blocks(A,[1, 2, 2]) 
except (TypeError): 
    print("ValueError")      
    

# Sixth - if inpiut not tensorflow 
try: 
    test4  = view_as_blocks(A.eval(),[1, 2, 2]) 
except (TypeError): 
    print("ValueError")      
         