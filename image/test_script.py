#Test Script

#Import Necessary Libraries
import numpy as np
import tensorflow as tf
import sobel 
import skimage.io
import matplotlib.pyplot as plt


#We are considering only 1 of the channel of the input image to do edge detection, because the edges associated with all the channels will be same.
image = skimage.io.imread('test_images/katy.jpg', as_grey=True)
#Calling the Sobel function to do edge detection
result = sobel.Sobel(image)

#Plotting the edge map of the image
plt.imshow(result[0,:,:,0], cmap="gray")
plt.show()
#Plotting the actual image
plt.imshow(image, cmap="hot")
plt.show()
