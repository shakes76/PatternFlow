# Richardson-Lucy deconvolution
_Author: Lachlan Paulsen
_Last update: 18/10/2019_

Richardson-Lucy deconvolution takes a blurred image with a known point spread function (psf). The algorithim makes a series of iterative convolutions in order to deblur the image.
To use the function simply pass in the image as a Tensor as well as the point spread function as a Tensor, the number of iterations to perform (More iterations will result in a smoother image but will increase computation time) and whether to clip absolute values larger than 1.

To run requires tensorflow-gpu, numpy, scikit-image, scipy and CUDA 10.0 (https://developer.nvidia.com/cuda-10.0-download-archive)


