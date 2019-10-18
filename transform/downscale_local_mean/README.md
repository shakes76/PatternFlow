# Module: Scikit- Image  - Transform  


## Downscale Local mean (downscale_local_mean) 

### Description of the Algorithum: 
Down-sample N-dimensional image by local averaging. The image is padded with `cval` if it is not perfectly divisible by the
integer factors.This function calculates the local mean of elements in each block of size factors in the input image.

Additonally, this algorithum uses - *view_as_blocks* algorithum, 
which is a block view of the input n-dimensional array that uses re-striding from numpy. 
In this algorithum, everything apart from re-striding is coded in Tensorflow
 
### How it works
* Paremeters: 
    * image : ndarray - N-dimensional input image 
    * factors : array_like - Array containing down-sampling integer factor along each axis.
    * cval :  Constant padding value if image is not perfectly divisible by the integer factors.
    * clip : bool, optional - Unused, but kept here for API consistency with the other transforms
        in this module.
* Output: 
    * image : ndarray - Down-sampled image with same number of dimensions as input image.
* Algorithum working: 
    * Check if factors is a tuple 
    * Check if all factor is greater than 1 
    * Check if the lenght of factor is same as shape of the image 
    * Get the padding of the image 
    * Get image, for new shape 
    * Use view_as_block to get the new array 
    * Use tf.reduce mean, to get final output 

### Dependencies 
The following libraries are used:
    * Tensorflow 
    * view_as_blocks 
        * as_strided (from numpy)

### Example of uses  
import numpy as np 
from downscale_local_mean import downscale_local_mean 
image = np.arange(15).reshape(3, 5)
downscale_local_mean(a, (2, 3))  
 





 