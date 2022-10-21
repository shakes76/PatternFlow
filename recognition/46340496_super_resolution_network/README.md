# Brain MRI Super-Resolution Network

## Description of algorithm and problem it solves
The algorithm it uses is Efficient Sub-Pixel Convolutional Neural Network (ESPCN). How this works
is it transitions a downscaled low-resolution image to a high-resolution image
by using sub-pixels instead of interpolation. The problem that this solves includes the following:
- HDTV
- Medical imaging
- Satellite imaging
- Face recognition
- Surveilance

## How it works
### get_model()
- Input: A low-resolution downscaled image is placed into the model, the reason for this is becasue
ESPCN doens't need the low-resolution image to be upscaled, as it enchances the image by using 
non-linear convolution to extract the features of the low-resolution image. Using the 
downscaled image then reduces the computational and memory complexity of the model.
- Convolution layers: These layers compose of L-1 layers which include the feature maps of the lowresolution image. This is usually done in two convolution layers but I added another one, with the relu activation to achieve better performance in the model with fewer epochs. These feature maps are used for helping with the high-resolution output
- Periodic Shuffling: Instead of shuffling the output as part of the sub-pixel convolution layer, it is done before the final layer so it can shuffle the training data to match the output layer which increased the speed of the model. This layer is instead of using a deconvolution layer and is log(base 2)r(squared) times faster. While r(squared) faster than any other implementations of upscaling before convolution.
- Sub-pixel convolution layer: This is the final layer of the model, this layer is what produces the
high-resolution image by using an upscaling filter for each of the feature maps produced in the previous convolution layers. The pixel-wise mean squared error (MSE) is then calculated for the reconsturction image.


### model.fit()
- ESPCN callback: This callback doesn't contribute anything to the training of the model. This class
just shows the psnr values between the high-resolution image and the prediction.

A visual representation of how Efficient Sub-Pixel Convolutional Neural Networks works
![A visual representation of how Efficient Sub-Pixel Convolutional Neural Networks works](./images/Figure%20of%20how%20ESPCN%20works.PNG)
Reference: https://arxiv.org/pdf/1609.05158.pdf?fbclid=IwAR0i0DMmECzsgRMQ0mYMWFQ3oxD3JkKty_3o7QW53HTV_qPHIIXqLn8bhRA

## Dependencies and Versions
- python - version: 3.9
- cudatoolkit - version: 11.2
- cudnn - version: 8.1.0
- tensorflow - version: 2.10.0
- matplotlib - version: 1.0
- numpy
- math

## Reproducibility of results
This isn't applicable as the results are very reproducable

## Inputs

### Input 1
![The first input of the low-resolution image](./images/Input%201%20for%20ESPCN.png)

### Input 2
![The second input of the low-resolution image](./images/Input%202%20for%20ESPCN.png)


## Outputs

### Output 1
![the first output of low-resoulution, high-resolution and the prediction](./images/Output%201.png)

PSNR value between high-resolution and low-resolution: 26.4325
PSNR value between high-resolution and prediction: 26.674484

### Output 2
![the second output of low-resoulution, high-resolution and the prediction](./images/Output%202.png)

PSNR value between high-resolution and low-resolution: 26.482525
PSNR value between high-resolution and prediction: 27.043264

### Output 3
![the third output of low-resoulution, high-resolution and the prediction](./images/Output%203.png)

PSNR value between high-resolution and low-resolution: 27.954971
PSNR value between high-resolution and prediction: 28.035522

### Output 4
![the fourth output of low-resoulution, high-resolution and the prediction](./images/Output%204.png)

PSNR value between high-resolution and low-resolution: 26.858969
PSNR value between high-resolution and prediction: 27.250572

### Output 5
![the fifth output of low-resoulution, high-resolution and the prediction](./images/Output%205.png)

PSNR value between high-resolution and low-resolution: 26.207376
PSNR value between high-resolution and prediction: 26.452433


## Plots

### PSNR
The PSNR value throughout 150 epochs, x-axis = epochs, y-axis = PSNR
![The PSNR value throughout 150 epochs, x-axis = epochs, y-axis = PSNR](./images/PSNR%20graph.png)

### Loss
The loss and validation loss
![The loss and validation loss](./images/loss%20and%20val%20loss.png)

## Pre-processing used and References
I personally didn't need to do any preprocessing on my images as the dataset I 
downloaded from https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download already
preprocessed the data.

The paper on ESPCN: https://arxiv.org/pdf/1609.05158.pdf?fbclid=IwAR0i0DMmECzsgRMQ0mYMWFQ3oxD3JkKty_3o7QW53HTV_qPHIIXqLn8bhRA

## Justification of Training, Validation and Testing Splits

### Training and Validation
I used a validation split of 20%, meaning 80% training and 20% validation, becuase I thought a good number for validation would be 20% of the entire dataset. This ensures that there is enough images to train the model, which then can be validated by a number of images.

### Testing Split
The testing split was given to me as a dataset by the preprocessed images that I downloaded. So I didn't set a testing split. 
