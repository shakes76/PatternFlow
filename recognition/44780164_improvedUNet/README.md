# Image Segmentation of the ISIC 2018 Dataset using an improved UNet

## Dataset

The dataset the network was trained on is the [ISIC 2018 challenge task](https://challenge2018.isic-archive.com/).

## Description

The problem the algorithm attempts to solve is detection of melanoma from images of people's skin. The algorithm works
by downsampling the 2D image using conv2D layers and then upsampling it to the correct size. This results in a 'U' shaped
network, hence the name.

This specific UNet is based on [this implementation](https://arxiv.org/pdf/1802.10508v1.pdf).

## Dependencies

* Python 3.7
* Tensorflow 2.3.1
* OS

## Usage

To run the driver script, place the data folder containing 2 folders, image and mask, into the same directory as the 
driver script.

