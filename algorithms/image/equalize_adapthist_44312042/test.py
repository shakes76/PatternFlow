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

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def plot_comparison(image, expected, actual):
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 3, 1)
    for i in range(1, 3):
        axes[0, i] = fig.add_subplot(2, 3, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 3):
        axes[1, i] = fig.add_subplot(2, 3, 4+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(image, axes[:, 0])
    ax_img.set_title('Original Image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(expected, axes[:, 1])
    ax_img.set_title('Skimage CLAHE')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(actual, axes[:, 2])
    ax_img.set_title('Tensorflow CLAHE')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()


def compare(expected, actual):
    return (expected==actual).all()
    
# Load an image to use for testing
test_image = data.moon()
expected_result = exposure.equalize_adapthist(test_image, clip_limit=0.03)

#plot_comparison(test_image, expected_result, expected_result)
print(compare(expected_result, expected_result))
#plt.show()
import equalize_adapthist

a, b = equalize_adapthist.histogram(test_image)

equalize_adapthist.tfhist(test_image)
print(a)
print(b)


# get equalization 

