
from skimage import data,img_as_float
from histogram import histogram
from matplotlib import pyplot as plt

import numpy as np


if __name__ == "__main__":
    image = img_as_float(data.camera())
    # print(image.shape)

    ##ORIGINAL IMAGE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    ax1.set_title("Original")
    ax1.imshow(image)

    # ##NUMPY HISTOGRAM
    np_hist,np_bins = np.histogram(image, bins=256)
    # (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    
    ax2.set_title("Numpy Histogram\n")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Pixels")
    width = 0.7 * (np_bins[1] - np_bins[0])
    center = (np_bins[:-1] + np_bins[1:]) / 2
    ax2.bar(center, np_hist, align='center', width=width)
    
    #ax3.imshow(np_hist, aspect='auto')

    ##TENSORFLOW HISTOGRAM
    h = histogram(image)
    hist,bins_center = h
    print(hist, bins_center)
    # (array([107432, 154712]), array([ 0.25,  0.75]))

    ax3.set_title("Histogram\n")
    ax3.set_xlabel("Pixel Value")
    ax3.set_ylabel("Pixels")
    width = 0.7 * (np_bins[1] - np_bins[0])
    #ax3.imshow(h, aspect='auto')
    ax3.bar(bins_center, hist, align='center', width=width)
    

    #fig.tight_layout()
    plt.show(block=True)
    
    fig.savefig('test.png')


        
