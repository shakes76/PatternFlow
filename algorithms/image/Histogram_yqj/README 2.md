# Algorithm of skimage.exposure.histogram by tensorflow

The algorithm is used to analyse the color of image, it will return a histogram of the numbers of pixels in order to each bin of pixel values.The histogram is working on the flattened image; for the color images, the function should be used on each color channel respectively and return the histogram of each color.

The function **histogram** has four parameters, includes image, nbins, source_range, normalize. The parameter 'image' should be 8-bit unit numpy array. 'nbin' represents to the number of bins of the histogram, here we should set to 256, since we have 256 different values of pixels. And we set 'source_range' as 'image', and normalize means TRUE or FALSE the image is normalised.In this function, the first step is to transfer the image from array to tensor. Second step is to reshape the tensor to one column. Third step is to calculate the number of pixels in each bin of histogram (y-axis), and set the x-axis of the output histogram. While another file main.py is to call the histogram function and plot both picture and its histogram.

The below is the output of the algorithm:

![Image text](https://github.com/QijiangYao/PatternFlow/raw/topic-algorithms/Histogram_yqj/output.PNG)

Example usage: image quality enhancement such as image smoothing, which is used to Reduce or eliminate mutations, edges, and noise in the image.
