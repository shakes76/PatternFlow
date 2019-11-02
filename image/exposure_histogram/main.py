
from skimage import data,img_as_float
from histogram import histogram
from matplotlib import pyplot as plt

import numpy as np

"""
    COMP3710 Report - Algorithm Implementation

    Student: Umberto Pietroni 45981427

    Test Drive Script which calls numpy histogram function 
    and tensorflow histogram function and plot them

"""

if __name__ == "__main__":

    #Get the image
    #image = img_as_float(data.camera())
    image = data.camera()
    nbins=256


    ##ORIGINAL IMAGE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    ax1.set_title("Original")
    ax1.imshow(image)

    ##NUMPY HISTOGRAM
    np_hist,np_bins = np.histogram(image, bins=nbins)

    ax2.set_title("Numpy Histogram\n")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Pixels")
    center = (np_bins[:-1] + np_bins[1:]) / 2
    ax2.bar(center, np_hist, align='center')


    ##TENSORFLOW HISTOGRAM
    hist,bins_center = histogram(image)
    print(hist, bins_center)

    ax3.set_title("Tensorflow Histogram\n")
    ax3.set_xlabel("Pixel Value")
    ax3.set_ylabel("Pixels")
    ax3.bar(bins_center, hist, align='center')
    
    fig.tight_layout()
    plt.show(block=True)
    fig.savefig('resources/exposure-histogram.png')


        
