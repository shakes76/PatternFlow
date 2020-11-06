# Image Segmentation of ISICs dataset with Unet


### Dependencies
* Tensorflow-gpu 2.1
* Matplotlib
### Description
The aim of the project is to successfully perform lesion segmentation on the ISIC dataset. Classifying each pixel either black or white, the following algorithm achieves this by using Unet. Images are first resized so they are 512x512. We pick this particular size because any smaller and completely black images would return a high dice coefficient which we don't want. We then input the images into the Unet where we use 

### Training and Results
We train our model over 6 epochs, where we achieve
