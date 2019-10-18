# A 2D Gaussian filter function for Tensorflow

This project is an attempt to port the Gaussian filter algorithm from the [filtering module of Scikit-Image](https://scikit-image.org/docs/stable/api/skimage.filters.html) to Tensorflow. The source code of skimage.filters.gaussian(image[, sigma, …]) can be found [here](https://github.com/scikit-image/scikit-image/blob/v0.16.1/skimage/filters/_gaussian.py#L12). The algorithm is used to apply a multidimensional Gaussian filter to an image. This has the effect of blurring the image relative to the specifications of the Gaussian kernel. The code for the filter in the Scikit-Image module is relatively complex, but it can be done in very few lines with Tensorflow. The basic idea was to create a Gaussian kernel and convolve this kernel with the image to be filtered. This implementation will is not multidimensional, it will only work for two dimensions. 

### How it works



### Figures



## Example usage



### Comments
