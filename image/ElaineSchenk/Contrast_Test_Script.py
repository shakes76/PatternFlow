#get image to actually perform alg on
from skimage.exposure import adjust_sigmoid
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import cv2
from Contrast_Corrector_Source import Contrast_Sigmoid #the tf implementation

''' Import image and store pixels in numpy array. 
Start a tensorflow session to run the function Contrast_Sigmoid. 
This is a tensorflow implemented version of the adjust_sigmoid function 
from the exposure module of Skikit-Image. It essentially provides contrast 
adjustment to a poorly contrasted image.'''
#import image:
array = cv2.imread("test_image.png")
#sensible defaults:
cutoff = 0.5
gain = 3
inv = False
#print out the adjust_sigmoid applied to image result. 



#Now run the tensorflow implemenation in a session. 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    corrected_array = Contrast_Sigmoid(array,cutoff,gain,inv).eval()

#plot tensorflow result
plt.subplot(1,3,1)
plt.imshow(array)
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(corrected_array)
plt.title("Tensorflow Corrected")


plt.show()
