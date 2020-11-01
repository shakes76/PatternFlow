# UNet Image Segmentation of the ISIC 2018 Challenge Dataset

## The Dataset
The ISIC 2018 Challenge tasked participants with developing image analysis tools to enable automated segmentation, classification and diagnosis of melanoma from dermoscopic images.  The challenge comprised of three tasks: Lesion Segmentation, Lesion Attribute Detection and Disease Classification.  This model aims to complete task 1: Lesion Segmentation.  An example of a lesion and its ground-truth segmentation and UNet generated segmentation is shown below.

// insert img here

## The Model
A UNet is a type of convolutional neural network which uses a U-shaped encoder-decoder structure.  It includes a contracting path (which follows the typical architecture of a convolutional network), followed by an expansive path which upsamples the pooled data back to its original shape.  A visualisation of the structure is shown below.
![UNet Structure](images/standard_unet.png)

# Dependencies

## To use the model.py file

* Python 3.7.9
* Tensorflow 2.1.0

## To run the driver.py script

* Python 3.7.9
* Tensorflow 2.1.0
* Scikit-learn 0.23.2
* Numpy 1.19.1
* Matplotlib 3.3.1
* Pillow 7.2.0

To run the driver script, the ISIC 2018 challenge data must be downloaded - the download link is <https://cloudstor.aarnet.edu.au/sender/?s=download&token=505165ed-736e-4fc5-8183-755722949d34>.  
There should be two folders of images, titled "ISIC2018_Task1_Training_GroundTruth_x2" and "ISIC2018_Task1-2_Training_Input_x2" - both need to be placed in the same directory as the driver.py file.

# Usage

## How to use model.py

### make_model()
Returns a Keras model with the UNet convolutional neural network structure.  The model follows the architecture shown by the image above - with a contracting and expansive path.  The model is compiled with the Adam optimiser, a binary cross-entropy loss function and uses the Dice similarity coefficient as a metric.

The contracting path has four repeated applications of two 3x3 unpadded convolutions (each followed by a ReLU), and a 2x2 max pooling operation.

The expansive path uses four repeated applications of two 3x3 unpadded convolutions with ReLU activation functions, followed by a 2x2 upsampling operation.  After the four applications, a final 1x1 convolution with a softmax activation function is applied to statistically categorise the binary value of each pixel for the output.
