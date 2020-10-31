# UNet Image Segmentation of the ISIC 2018 Challenge Dataset

## The Dataset
The ISIC 2018 Challenge tasked participants with developing image analysis tools to enable automated segmentation, classification and diagnosis of melanoma from dermoscopic images.  The challenge comprised of three tasks: Lesion Segmentation, Lesion Attribute Detection and Disease Classification.  This model aims to complete task 1: Lesion Segmentation.

## The Model
A UNet is a type of convolutional neural network which uses a U-shaped encoder-decoder structure.  It includes a contracting path (which follows the typical architecture of a convolutional network), followed by an expansive path which upsamples the pooled data back to its original shape.
![](images/standard_unet.png)
