# Histogram
Histogram of image implemented for Tensorflow.

This function is created and maintained by Puyang Deng. StudentID: s4443048.

## Description
Return histogram of image.
The histogram function returns the centers of bins and does not rebin integer arrays. 

For integer arrays, bins woulld counted by the function _bitcount_histogram in tensorflow:

* Since each integer value has its own bin, which improves speed and intensity-resolution. The function also contain an _offset_array function to offset the array to get the lowest value at 0 if negative if _bitcount_histogram applied.

For image array not consisted of integer, the function would:
* Define the histogram range as [0.0,256.0] for image source.
* Using the dtype_limits function to get the intensity limits, i.e. (min, max) tuple based on the dtype.
* Compute histogram on the image converted into tensor: for color images.
the function should be used separately on each channel to obtain a histogram for each color channel.

## How it works

The function works with the following examlpe using skimage data:

        import matplotlib.pyplot as plt

        from skimage.util import img_as_ubyte
        from skimage import data
        from histogram import *

        noisy_image = img_as_ubyte(data.camera())#import image as ubyte
        hist, hist_centers = histogram(noisy_image)#apply the function

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

        ax[0].imshow(noisy_image, cmap=plt.cm.gray)
        ax[0].axis('off')

        ax[1].plot(hist_centers, hist, lw=2)
        ax[1].set_title('Histogram of grey values')

        plt.tight_layout()
        
## Output

  ![Dapr overview](https://github.com/craigdeng/PatternFlow/blob/topic-algorithms/image/histogram_s4443048_Puyang_Deng/output.png)
        
