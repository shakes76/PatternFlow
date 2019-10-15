import sys

import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.misc

from impl import histogram

def get_greyscale_face():
    face = scipy.misc.face()
    grey = tf.image.rgb_to_grayscale(face)
    return grey[:,:,0]

def show_histogram(image):
    values, centers = histogram(image, normalize=True)
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].imshow(image.eval(), cmap=plt.cm.gray)
    ax[0].axis('off')

    ax[1].plot(centers.eval(), values.eval(), lw=2)
    ax[1].set_title('Histogram of grey values')

    plt.tight_layout()


if __name__ == '__main__':
    show_histogram(get_greyscale_face())