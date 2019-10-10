
from skimage import data,img_as_float
from histogram import histogram

import numpy as np
if __name__ == "__main__":
    # image = mpimg.imread('phantom.png')
    image = img_as_float(data.camera())
    print(image.shape)
    p = np.histogram(image, bins=2)
    print(p)
    # (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    h = histogram(image, nbins=2)
    print(h)

    # (array([107432, 154712]), array([ 0.25,  0.75]))