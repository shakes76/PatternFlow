try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf 
from 42370057_algorithm import *

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
    intensity_test = tf_rescale_intensity(image) 
    ax = plt.subplot(1,2,2)
    plt.imshow(intensity_test)
    plt.show()

main()