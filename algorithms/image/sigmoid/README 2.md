# Yitong Dai's assignment 3 of COMP3710
Implementation of sigmoid correction from the exposure module of Scikit-Image

## Description of algorithm
Known as Contrast Adjustment. This function transforms the input image pixelwise according to the equation 

__ O = 1/(1 + exp*(gain*(cutoff - I))) __ after scaling each pixel to the range 0 to 1.
  
- Parameters:
	1. input_img : _ndarry_  
	
				input image
				
	2. cutoff : _float optional_ 
	
			Cutoff of the sigmoid function that shifts the characteristic curve in horizontal direction.
			Default value is 0.5.
			
	3. gain : _float optional_
	
			The constant multiplier in exponential’s power of sigmoid function.
			Default value is 10.
			
	4. inv : _bool optional_
	
			If True, returns the negative sigmoid correction. 
			Defaults to False.
			
- Returns:

	1. out : _ndarray_
	
			Sigmoid corrected output image.
		
The function begins by importing the tensorflow library, then I transform the input_img to a tensor and normalize it.
After that preparation, I calculate according to the equation to get the result, and turn it back to original mode. Then we 
run the computation through the interactive mode which is efficient to get immediate execution. During the running step, we 
initialise the graph which is immportant. Finally, we return the result we get.

## Sigmoid function and how it works
In the image enhancement, the objective of it is dependent on the application circumstances, and contrast is an important factor in any individual
estimation of image quality. It can be a controlling tool for documenting and presenting information collected during examination, and also can improve
the image appearance by increasing dominance of some features or by decreasing ambiguity between different regions of the image.


Sigmoid function is a continuous non-linear activitation function. The name, sigmoid derives from the fact that the function is "S" shaped, so this 
function resemble S-shaped curves.


Adaptive sigmoid function step:

It's a It is a point process approach that is performed directly on each pixel of an image,independent of all other pixels in the image to
change the dynamic range.In this the mask that is applied to the target images is a non-linear activation function, which is called sigmoid 
function multiplied by input itself and by a factor. We used that factor to determine the most wanted degree of the contrast depending on the 
degree of darkness or brightness of the original image.

A pixel value in the enhanced window dependents only on its value that means if the variance of window pixels is less than variance of the 
image and greater than the global variance then the value of pixel under consideration is remapped and if the interest pixel does not satisfy 
the condition its value remains unchanged.

Where gain is a contrast factor determines the degree of the needed contrast. The value of gian depends on the objective of the enhancement 
process,the user can select the value of gain according to desired contrast that they needs.
		
The pixel values are within a limited range (0-255) for an 8-bit image.The results usually needs to be clipped to the maximum and minimum
allowable pixel values so that all highest components turn out to be 255 and lowest values to 0. That's why we need to normalize the input and turn
it back after calculation.

After map window reaches theright side, it returns to the left side and moves down a step. The process is repeated until the sliding window 
reaches the right-bottom corner of the image.
	
## Performace of evalution
original image :

![Image of Wally](https://github.com/Lynn-Dai/PatternFlow/blob/master/image/sigmoid/Wally.jpg)

adjusted image :

![Image of Adjusted_Wally](https://github.com/Lynn-Dai/PatternFlow/blob/master/image/sigmoid/adjust_wally.jpg)

original grey image :

![Grey image](https://github.com/Lynn-Dai/PatternFlow/blob/master/image/sigmoid/dark_origin.jpg)

adjusted grey image :

![Adjusted grey image](https://github.com/Lynn-Dai/PatternFlow/blob/master/image/sigmoid/adjust_dark.png)


## Conclusion

The results obtained varied depending on the processed images, but we get a much higher readability than its origin form and prove highly effective
in dealing with poor contrast images, and can easily improve the contrast "darkness and the brightness", so the process of image segmentation and
classification can be powerfully accomplished from the resulting images. For the future thing, we can adjust the modulus to decrease the noise present 
in the smooth areas to perform the image in the better condition.
	
## Reference

[1] Naglaa Hassan and Nario Akamatsu, "A New Approach for Constrast Enhancement Using Sigmoid Function", '<The International Arab Journal of Information
Technology, Vol.1, No.2, July 2004'>

[2] Hui Zhu, Francis H. Y. Chan, and F. K. Lam,” Image Contrast Enhancement by Constrained Local Histogram Equalization”, Computer Vision and Image 
Understanding Vol. 73, No. 2, February, pp. 281–290, 1999.

[3] Saruchi, "Adaptive Sigmoid Function to Enhance Low Contrast Images", International Journal of Computer Applications (0975 – 8887) Volume 55– No.4, 
October 2012
