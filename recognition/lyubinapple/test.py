'''
    File name: test.py
    Author: Bin Lyu
    Date created: 10/23/2020
    Date last modified: 
    Python Version: 4.7.4
'''
import tensorflow as tf
import matplotlib.pyplot as plt
# filters number, kernel_size(F), strides(S)
# conv: ((W-F+2P)/S)+1. Pooling:((W-F)/S)+1
a = (10, 64, 64, 3)
x = tf.random.normal(a)
print(x)
