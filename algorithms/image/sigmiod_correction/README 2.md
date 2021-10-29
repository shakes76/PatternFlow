# Adjust_sigmoid algorithm reimplementation in TensorFlow
Reimplement the adjust_sigmoid function of Scikit-image exposure module

## Description of algorithm
Also known as Contrast Adjustment. This function transforms the input image pixelwise according to the equation O = 1/(1 + exp*(gain*(cutoff - I))) after scaling each pixel to the range 0 to 1.
  
* __Parameters:__
	1. __input_img : ndarry__  
	
			input image
				
	2. __cutoff : float optional__ 
	
			Cutoff of the sigmoid function that shifts the characteristic curve 
			in horizontal direction. Default value is 0.5.
			
	3. __gain : float optional__
	
			The constant multiplier in exponential’s power of sigmoid 
			function. Default value is 10.
			
	4. __inv : bool optional__
	
			If True, returns the negative sigmoid correction. Defaults to False.
			
* __Returns:__

	1. __out : ndarray__
	
			Sigmoid corrected output image.

* __Reference__

	[1] Gustav J. Braun, “Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions”, [http://www.cis.rit.edu/fairchild/PDFs/PAP07.pdf](http://www.cis.rit.edu/fairchild/PDFs/PAP07.pdf)

This function is only based on TensorFlow, which has similar capabilities with [adjust_sigmoid function in Scikit-image exposure module.](https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid)
This function only supports non-negative ndarray as the input image. If input is not >= 0 everywhere, message, as well as the first summarize entries of input are printed, and InvalidArgumentError is raised.
Also, images are simply numpy arrays, which support a variety of data types. To avoid distorting image intensities (for min-max normalization), I assume that images use the following dtype ranges:

Data type | Range
------------ | -------------
uint8 | 0 to 255
uint16 | 0 to 65535
uint32 | 0 to 2^32 - 1
float | 0 to 1
int8 | 0 to 127
int16 | 0 to 32767
int32 | 0 to 2^31 - 1

Note that float images should be restricted to the range 0 to 1 even though the data type itself can exceed this range.

## Principle of the sigmoid correction
Sigmoid function is a continuous non-linear function, which is "S" shaped. 
This enhancement approach is performed directly on each pixel of and image. 
In this method, the input image is multiplied by a non-linear activation function and by a factor.
For example, the pixel values are within a limited range (0-255) for an unit8 image.
The results usually need to be clipped to the minimum or maximum allowable pixel values so that all highest components turn out to be 255 and the lowest values to 0.

__Reference__

[2] Hassan1&2, N., & Akamatsu, N. (2004). A new approach for contrast enhancement using sigmoid function. [http://www.ccis2k.org/iajit/PDF/vol.1,no.2/10-nagla.pdf](http://www.ccis2k.org/iajit/PDF/vol.1,no.2/10-nagla.pdf)

## Example usage
Return the sigmoid transformed image
```python
import cv2
from sigmiod_correction.sigmoid import adjust_sigmoid
# try to test a real image
img = cv2.imread('uni.jpg')
# Call the sigmoid function.
result = adjust_sigmoid(img)
# Display the result.
cv2.imwrite('sigmoid_uni_img.jpg', result)
cv2.imshow("sigmoid_uni_img", result)
cv2.waitKey()
```
	
## Test result
__Original image__ :

![Image of University of Queensland](https://github.com/RoyUQ/PatternFlow/blob/topic-algorithms/image/sigmiod_correction/resources/uni.jpg)

__Transformed image__ :

![Transformed Uni Image](https://github.com/RoyUQ/PatternFlow/blob/topic-algorithms/image/sigmiod_correction/resources/sigmoid_uni_img.jpg)

By comparing the results, we found that the shadows and highlights has been improved. Also, most of the scene features have become more clearly and more interpretable to the human eye.
This function can give high contrast images.

## Author
Name: Jialuo Ding

Student number: 44732024

## Dependencies
* Python 3.6
* TensorFlow 1.12.0
* Numpy 1.16.0
* OpenCV-Python 4.1.1.26
