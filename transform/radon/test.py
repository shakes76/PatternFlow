import sys
from radon import *
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread

def main(arglist):
    image = imread("test.png", as_gray = True)
    image = tf.constant(image)
    
    transformed = radon(image)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(transformed)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])