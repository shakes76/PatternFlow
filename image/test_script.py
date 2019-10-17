#Test Script

#Import Necessary Libraries
import numpy as np
import tensorflow as tf
import sobel 
import skimage.io
import matplotlib.pyplot as plt


image = skimage.io.imread('test_images/katy.jpg', as_grey=True)
#Calling the Sobel function to do edge detection
result = sobel.Sobel(image)

#Plotting the edge map of the image
plt.imshow(result[0,:,:,0], cmap="gray")
plt.show()
#Plotting the actual image
plt.imshow(image, cmap="hot")
plt.show()
