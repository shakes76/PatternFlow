# EQUALIZED HISTOGRAM

_Author: Duc Minh Phan_ 

_Last update:17/10/2019_

The presented module computes the skimage exposure equalizer histogram function using pure-tensorflow library. 

## Introduction

Histogram equalisation is an image processing technique used to improve the contrast of an image. 
This function is for grey-scale image. If the user wants to use get equalised histogram on RGB image, please reduce its dimension to the desired color.


## Algorithm explanation

__Function tf_equalize_histogram(image, nbins)__

Input: _image array , number of bins_



Output: _ndarray of the same shape with the input image_

- First, we compute the histogram information the image. This information stores the frequency of all pixel levels. By default, the number of bins is 256 and these values are from 0 - 255.
- Second, we calculate the cumulative distribution function histogram. 
- Then, compute the new values from the cummulative function above.
- Last, get the shape of the image and assign the new values and return the equalised image.	

### Example usage

This is example code, that can be found in the driver script.
> image = cv2.imread(filename, 0)
> 
> eq_image = tf_equalise_histogram(image) 


### Result

Original Image
![orinal image]()

Gray Scale Image
![grey image]()

Equalised Histogram
![etf image]()






