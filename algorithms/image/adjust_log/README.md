# adjust_log


## Description
Performs Logarithmic correction on the input image totally with [Tensorflow](https://www.tensorflow.org/) 

## Princeples

- *O* = *gain*  * log2 (1 + **I**) 

- *O*= *gain* * (2^**I**)

A logarithmic transformation of an image is actually a simple one. We simply take the logarithm of each pixel value, and we’re done. 

## How it works
In the case of the following image, I simply take an input image, scale each pixiel value to [0, 1], calculate the base-2 logarithm of them (if "inv" is set to be True, calculate the 2^(scaled pixel value)), then normalize the image back to [0, 255], and round it to the nearest integer (note that the addition of a scalar 1 to prevent a log(0) calculation).

Besides, before calculating the logarithm values, this function check the will check the input to be non-negative which may return exception.


## Function

#### Parameters

>**image**: ndarray

> Input image.

>**gain**: float, optional

> A constant multiplier. Defalt value is 1.

>**inv**: bool, optional

> If inv is set to be True, it performs inverse logarithmic correction, else 
correction will be logarithmic. Defaults to False.

#### Returns
>**out**: ndarray

> Logarithm corrected output image.


## Dependencies

### Function requirement
> Tensorflow *version 1.14 or later*

### Test files environment
> matplotlib.pyplot



## Example results
After using the adjust_log function, I got the results like: 

![](https://github.com/qg007110/PatternFlow/blob/topic-algorithms/image/adjust_log/output.png) 
