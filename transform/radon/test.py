import sys
from radon import *
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread

def main(arglist):
    tf.compat.v1.disable_eager_execution()
    #tf.executing_eagerly()
    sess = tf.compat.v1.Session()
    image = imread("test.png", as_gray = True)
    image = tf.constant(image)
    
    #print(bilinear_interpolation(image, 256, 256, 125, 125, 5).eval(session=sess))
    
    transformed = radon(image)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.eval(session=sess))
    plt.subplot(1, 2, 2)
    plt.imshow(transformed.eval(session=sess))
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])