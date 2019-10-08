import sys
import radon
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage.transform as sk

def main(arglist):
    tf.compat.v1.disable_eager_execution()
    #tf.executing_eagerly()
    sess = tf.compat.v1.Session()
    image = imread(arglist[0], as_gray = True)
    tensorimage = tf.constant(image)
    
    transformed = radon.radon(tensorimage, list(range(60)))
    check = sk.radon(image, list(range(60)))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(transformed.eval(session=sess))
    plt.subplot(1, 3, 3)
    plt.imshow(check)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])