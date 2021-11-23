# Segmentation of MRI scans of the Prostate using UNet3D

## Dataset
The data comes from a series of MRI scans on patients undergoing a radiation therapy study. Each scan is in NifTi file format. Each case number is a different patient. Week 0 represents before treatment and each week after is another week in. The dataset can be found here: https://data.csiro.au/collection/csiro:51392v2.

## Pre-preparation
Each scan follows the 3D shape (256,256,128) but one. This scan, from Case 19 Week 1, needs to be removed from the dataset files for the model training to work.

This model will be using a train-test-split of 80/10/10; however, one patient's scans can't be in two different sets as that would cause data leakage and skew the validation or test results. Thus, there is a function to split based on patient number.

## UNet3D
UNet (2D) was developed for medical image segmentation, containing a normal convolution/max pooling path called the encoder and a transposed convolution path called the decoder. These two paths are linked by skip connections to allow for higher precision. This process enables the model to first learn what information exists in some dataset, then find where it is located with low memory usage and high accuracy.

UNet3D is simply a 3-dimensional implementation of this model, replacing the 2D layers with their 3D counterparts.


## Dependencies
This project uses the following libraries on their respective versions:
- Python 3.9.7
- Tensorflow 2.4.1
- Matplotlib 3.4.3
- Nibabel 3.2.1
  

## Usage
Download the dataset .zip file from the [CSIRO link](https://data.csiro.au/collection/csiro:51392v2). Extract the labels and MRIs folders into the empty folder named dataset.

Now the model can be trained and run with `python driver.py`.

When the driver file is run, it will generate a .h5 model file. If you don't wish to re-train the model, you can run `python restore_model.py {path_to_model.h5}` to re-fetch the test performance. 
 
## Examples
Running the model for 15 epochs gave train/test/validation dice scores of over 0.8. See below for the performance over this time:
![training and validation scores](https://github.com/borova-siska/PatternFlow/blob/topic-recognition/recognition/s4580429%20Prostate%20Unet3D/images/dice.png?raw=true)

Testing the model on a random slice of one of the test images gives the following comparison:
![Comparison between testing label and predicted labels](https://github.com/borova-siska/PatternFlow/blob/topic-recognition/recognition/s4580429%20Prostate%20Unet3D/images/compare.png?raw=true)