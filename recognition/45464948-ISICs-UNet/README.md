# Improved UNet : Segment the ISICs data set with the Improved UNet
The problem solved by this report is the binary segmentation of the ISICs melanoma data set. Divided into healthy and unhealthy.

Student Number: 45464948

Student Name: JINGYI LI

Course Number: COMP3710

## Algorithm
U-Net is a convolutional neural network. Unet consists of two parts. The first part is feature extraction. The second part is the up-sampling part. In other words, encoder and decoder.
![Improved UNet](https://github.com/unicorn10086/PatternFlow/blob/topic-recognition/recognition/45464948-ISICs-UNet/images/improvedunet.png)
This improved UNet is developed by F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein. As the standrad UNet, the improved UNet has two part, decoder and encoder.
>The activations in the context pathway are computed by context modules.Each context module is in fact a pre-activation residual block [13] with two 3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between[^1].

## Running program
Download the improvedUNet.ipynb and change the dataset address in your computer.
Or run the main.py.

## ISIC Dataset
The ISIC data set is a melanoma research data set. The data set used in this project is taken from 2018 data. Divided into input folder and groundtruth folder. There are 2594 jpg files in the input folder. There are 2594 png files in the groundtruth folder.

## Dice Similarity Coefficient
Dice Similarity Coefficient is a similarity function to evalute how similar between two images.
![Dice Similarity Coefficient](https://github.com/unicorn10086/PatternFlow/blob/topic-recognition/recognition/45464948-ISICs-UNet/images/dice.png)

## Example Output
Input skin images and ground truth, comparing with predicted images.
![plot1](https://github.com/unicorn10086/PatternFlow/blob/topic-recognition/recognition/45464948-ISICs-UNet/images/plot1.png)
![plot2](https://github.com/unicorn10086/PatternFlow/blob/topic-recognition/recognition/45464948-ISICs-UNet/images/plot2.png)
![plot3](https://github.com/unicorn10086/PatternFlow/blob/topic-recognition/recognition/45464948-ISICs-UNet/images/plot1.png)

## Split Data 
train/test/validation split 70:20:10  for 100 epochs

Train data set for training model. Validation data set for choosing parameter. Test data set for test the accuracy of mmodel. This ISIC dataset is big enough, there is no need to using data augmentation.

## Dependencies
Python3.8

tensorflow2.6.0

sklearn1.0

glob

matplotlib3.4.3

jupyter notebook

## Reference
[^1]: Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge

