#get image to actually perform alg on
from skimage.exposure import adjust_sigmoid
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

''' Import image and store pixels in numpy array. 
Start a tensorflow session to run the function Contrast_Sigmoid. 
This is a tensorflow implemented version of the adjust_sigmoid function 
from the exposure module of Skikit-Image. It essentially provides contrast 
adjustment to a poorly contrasted image.'''

#print out the adjust_sigmoid applied to image result. 



#Now run the tensorflow implemenation in a session. 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(Contrast_Sigmoid(Trial_Array_1)))

#plot tensorflow result
