# Super resolution CNN

Author: Matthew Griffiths, 24571417

## Super-Resolution CNN Algorthm
The SRCNN (Super-Resolution CNN) is a deep learning model that reconstructs a low-resolution 
image into a high-resolution version of that image. Using efficient "sub-pixel convolution" 
layers, the model learns an array of image upscaling filters. The implementation here utilises 
three layers and takes input as an image that has been downsampled by a factor of 4. 

The dataset utilised is the ADNI MRI brain scan dataset.

## The Problem
SRCNN's have many applications, typically within imaging. Some examples include surveillance 
for performing facial recognition on low resolution images, upscaling low resolution media files 
for reducing server costs and generating high resolution MRI images for medical purposes.

