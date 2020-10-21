# -*- coding: utf-8 -*-
# Author: Yuzhe Jie
# Last update: 18/10/2019

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.data import camera
from skimage.filters import sobel_h

import my_sobel_h

##Driver script
if __name__ == "__main__":
    image = camera()
    # the image filtered by skimage.filters.sobel_h
    edge_sobel_h = sobel_h(image)
    # the image filtered by my_sobel_h
    result = my_sobel_h.tf_sobel_h(image)
    # initialize tensorflow variables
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    filter_result = result.eval()
    # Here does some masking, for the outliers of the image
    filter_result[0, :] = 0
    filter_result[-1, :] = 0
    filter_result[:, 0] = 0
    filter_result[:, -1] = 0
    # plot the images filtered by skimage.filters.sobel_h
    # and by my implemented tf_sobel_h
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))

    ax[0].imshow(edge_sobel_h, cmap=plt.cm.gray)
    ax[0].set_title('Sobel_h Edge Detection')

    ax[1].imshow(filter_result, cmap=plt.cm.gray)
    ax[1].set_title('my Sobel_h Edge Detection')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()