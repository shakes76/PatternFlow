# Tensorflow Implementation of  Improved UNET

## Description of Algorithm & The ISIC Dataset

The Improved UNet is a CNN originally developed for biomedical image segmentation in order to produce a mask for skin lesion images. This algorithm ultimately contributes to medical research by improving the diagnosis of melonoma in patients.

The ISIC dataset is a open source dataset seeking to advance melonoma research. Each year they run the ISIC Challenge which aims to improve diagnosis accuracy using AI. The dataset used in this model is taken from the 2018 challenge.


## Architecture
The Improved UNet Architecture is displayed below. The model works by continously encoding the data in higher abstract forms through convolution, before recombining the representations to identify features used to form a segmentation mask. Like the standard UNET, the Improved UNet model can thought of as containing two separate parts - encoding and decoding.
<p align="center"><img src='images/improved_unet.png'></p>

The model was trained with a 80:10:10 train/test/validation split for 30 epochs to achieve a >0.8 dice similarity. The data was preprocessed through normalization and resizing to 256x256 with 3 channels for input data, and 1 channel for the mask data.


### Google Colab

This model has been developed using Google Colab. Henceforth, all of the logic (such as mounting drives) assumes that setup. There are some preliminary steps if you would like to run it within that environment.

1. You will need to upload the dataset to your Google Drive, along with (optionally) model.py. This could take acouple of hours depending on your internet speeds.
2. You will also need to update the directory strings to be the relative location of your dataset

### Running on Personal Computer

If you would like to run this on your computer then please just do not run the first cell of main.ipynb. Everything else should work accordingly (as long as you update the path to the dataset).

## Results

### Loss Function and Accuracy Plot

A Dice Similarity Loss function is used as we are aiming for a Dice similarity of over 0.8. The Dice Coefficient was chosen as the metric to evaluate the model as it is a straightforward mechanism used to evaluate the similarity of two images. A higher value represents higher similarity. 
<p align="center"><img src='images/training_info.png'></p>


### Data Visualisation

The below image demonstrates the model predicting the mask of 3 different skin lesion images.
<p align="center"><img src='images/visualisation.png'></p>

### Dependencies 

## Model Dependencies
- Python 3.7
- tensorflow

## Driver Dependencies 
- Python 3.7
- sklearn
- glob
- tensorflow
- numpy
