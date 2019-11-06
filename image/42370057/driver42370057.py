
"""
Author: Cameron Gordon, 42370057 
Licence: GNU General Public Licence v3.0
File provided free of copyright 
Date: 6 Nov 2019

Two algorithms from the Scikit Exposure module implemented in Tensorflow.

tf_intensity_range implements the intensity_range function
tf_rescale_intensity implements the rescale_intensity function
The algorithms are contained in the algorithm42370057.py file. 

The below code is the driver module. 
It calls an example image face and performs the tf_rescale_intenstiy function, plotting the output. """

import tensorflow as tf 
from algorithm42370057 import *

print(tf.version)

import matplotlib.pyplot as plt 
from scipy import misc
# uses the default scipy raccoon face as the example image 

def main(): 
    """
    Author: Cameron Gordon, 42370057
    Performs a test using the scipy raccoon face as an example image.
    Then calls intensity_test, which is then plotted. 
    """ 
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(1,2,1)
    face = misc.face()
    plt.imshow(face)
    
    image = face # example image 
    intensity_test = tf_rescale_intensity(image) # rescales using the test image 
    ax = plt.subplot(1,2,2)
    plt.imshow(intensity_test)
    plt.show()

main()
