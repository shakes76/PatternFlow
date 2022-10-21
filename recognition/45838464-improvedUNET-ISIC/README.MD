# Skin Mole Segmentation on the ISIC2017 data set using the improved UNet.

## Author
Name: Michael Smith

## Problem Description
Image segmentation is important in the current scene of medicine as it allows patients gain important information without the immediate precense of a doctor. This is especially beneficial for people who don't have reliable access to a hospital or GP. Image segmentation is the process of separating an image into its constituent classes. In this exampe the two classes are the skin and the mole.   

## Why UNet
The improved UNet model provides boosts in performance over the original UNet model developed by THESE GUYS, through small tweaks in the architecture. 


## How it works
The UNet architecture is an AutoEncoder CNN which means that it takes data, encodes that to a low dimension latent space, then decodes the latent space to a segmentation. 


## Examples

## Dependecies
python 3.7.13
tensorflow 2.9.1
matlplotlib 3.5.2
numpy 1.21.6
glob2 0.7
opencv 4.6.0