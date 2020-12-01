"""
Used to test tensorflow implementation of image procssing algorithm

sklearn reuslts will be used to validate outputs
some code used from scikit-image example of equalize_adapthist, to ensure
testing is representiative of intended use.
https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
"""

#import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from skimage import data, img_as_float
from skimage import exposure

matplotlib.rcParams['font.size'] = 8

test_image = data.moon()
expected_result = exposure.equalize_adapthist(test_image, clip_limit=0.03)

import equalize_adapthist

a, b = exposure.histogram(test_imaGe, nbins=20)

c, d = equalize_adapthist.histogram(test_image, nbins=12)

print(a, b, c, d)
# get equalization 

