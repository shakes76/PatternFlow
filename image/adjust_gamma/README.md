# Module: Scikit- Image  - Exposure   

## Adjust Gamma (adjust_gamma) 

### Description of the Algorithum: 
This algorithum is also known as Power Law Transform. This function transforms the input image pixelwise according to the
equation ``Output = Input**gamma`` after scaling each pixel to the range 0 to 1.

* For gamma greater than 1, the histogram will shift towards left and the output image will be darker than the input image.
* For gamma less than 1, the histogram will shift towards right and the output image will be brighter than the input image. 

### How it works
* Parameters: 
    * image : ndarray - Input image
    * gamma : float, optional -  Non negative real number. Default value is 1.
    * gain : float, optional - The constant multiplier. Default value is 1.

* Outputs: 
    * ndarray, Gamma corrected output image. 

* Algorithum working: 
    * Define a dtype_range dictionary, containing the range for every datatype 
        * For instance, for float64, range is (-1, 1)
    * Get the data type for image 
    * Get the max and min value of the data type of the image using dtype_range 
    * Check if max or min is less than zero, if so, clip it at zero 
    * Check if gamma is greater than zero 
    * Convert image to tensor, check if the image is greater than zero 
    * Calculate the output as ((image / scale) ** gamma) * scale * gain
    
### Dependencies 
The following libraries are used: 
    * Tensorflow 

### Example of uses  
from adjust_gamma import adjust_gamma 
from skimage import data, exposure, img_as_float 

image = img_as_float(data.moon()) 
output = adjust_gamma(image, 2)  
print(output)


![Actual Imafe](https://i.ibb.co/592wp2X/image.png) 
![Gamma adjust greater than 1](https://i.ibb.co/qkfFNbZ/output-gamma.png)
![Gamma adjust lesser than 1](https://i.ibb.co/xmnb1V0/3.png) 


