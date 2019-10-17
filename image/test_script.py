#Test Script

#Import Necessary Libraries
import numpy as np
import tensorflow as tf
import sobel 
import skimage.io
import matplotlib.pyplot as plt


image = skimage.io.imread('G:/IIT_MADRAS_DD/Semesters/7th sem (UQ)/COMP3710 (Pattern Recognition and Analysis)/Lab/Lab_3/PatternFlow/image/test_images/katy.jpg', as_grey=True)
result = sobel.Sobel(image)

plt.imshow(result[0,:,:,0], cmap="gray")
plt.show()
plt.imshow(image, cmap="hot")
plt.show()
