# UNet Image Segmentation of the ISIC 2018 Challenge Dataset

## The Dataset
The ISIC 2018 Challenge tasked participants with developing image analysis tools to enable automated segmentation, classification and diagnosis of melanoma from dermoscopic images.  The challenge comprised of three tasks: Lesion Segmentation, Lesion Attribute Detection and Disease Classification.  This model aims to complete task 1: Lesion Segmentation.

## The Model
A UNet is a type of convolutional neural network which uses a U-shaped encoder-decoder structure.  It includes a contracting path (which follows the typical architecture of a convolutional network), followed by an expansive path which upsamples the pooled data back to its original shape.
![](images/standard_unet.png)


# Dependencies
## To use the model.py file:
* Python 3.7.9
* Tensorflow 2.1.0
## To run the driver.py script:
* Python 3.7.9
* Tensorflow 2.1.0
* Scikit-learn 0.23.2
* Numpy 1.19.1
* Matplotlib 3.3.1
* Pillow 7.2.0

To run the driver script, the ISIC 2018 challenge data must be downloaded - the download link is https://cloudstor.aarnet.edu.au/sender/?s=download&token=505165ed-736e-4fc5-8183-755722949d34.  
There should be two folders of images, titled "ISIC2018_Task1_Training_GroundTruth_x2" and "ISIC2018_Task1-2_Training_Input_x2" - both need to be placed in the same directory as the driver.py file.
