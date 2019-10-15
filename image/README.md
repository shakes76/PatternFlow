# Algorithm of skimage.exposure.histogram

The algorithm is ued to analyse the color of image, it will return a histogram of the numbers of pixels in order to each bin of pixel values.The histogram is working on the flattened image; for the color images, the function should be used on each color channel respectively and return the histogram of each color.

The function **histogram** has four parameters, incluses image, nbins, source_range, normalize. The parameter 'image' should be 8-bit unit numpy array. 'nbin' represents to the number of bins of the histogram, here we should set to 256, since we have 256 different values of pixels. And we set 'source_range' as 'image, and normalize means TRUE OR FALSE the image is normalised.
