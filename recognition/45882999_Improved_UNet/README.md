# Improved UNet on ISICs data set
## Author
Name: Kieran Plume

Student number: 45882999

This project was completed for COMP3710

## Description
The algorithm created is an improved UNet model for the ISICs data set. This is derived from the UNet model which is made of an encoder-decoder in a U shape. This is to analyse segments of brain tumors for faster and more objective results.

## How it works
The context modules allows the more complex representations to be encoded as the network progresses down.
The localization modules are used to decode which takes information from low levels of the network and then upsample them to the higher levels of the network. These representations are combined to give a higher resolution.
### Context module
A context module contains two 3x3 convolutional layers with a dropout of 0.3 in between
### Upsampling Module
The layer is upscaled followed by a 3x3 convolutional layer
### Localization Module
The localization module contains a 3x3 convolution followed by a 1x1 convolution
### Activation
All modules use a LeakyReLU activation layer

Here you can see the summary of the model with the accuracy as 0.93 after 15 epochs

<img src="./images/accuracy.PNG" width="400"/>
<img src="./images/loss.PNG" width="400"/>

## Dependencies
* Python 3.7
* Tensorflow 2.1
* Tensorflow addons 0.9.1
* Matplotlib

## Dataset split
Seeing as the dataset was quite small 30% was allocated for testing and validation. This allowed for testing to give accurate results. As this took some data away from the training set it makes the results more accurate as there is less data.