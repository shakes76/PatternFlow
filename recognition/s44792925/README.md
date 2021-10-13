# Tensorflow Implementation of  Improved UNET

## Description of Algorithm & The ISIC Dataset

The Improved UNet is a CNN originally developed for biomedical image segmentation which is able to produce a segmentation mask for skin lesion images. This process will contribute to medical research by improving the diagnosis of melonoma in patients.

The ISIC dataset is a open source dataset seeking to advance melonoma research. Each year they run the ISIC Challenge which aims to improve diagnosis accuracy using AI. The dataset used in this model is taken from the 2018 challenge.


## Architecture
The Improved UNet Architecture is displayed below. The model works by continously encoding the data in higher abstract forms through convolution before recombining the representations to identify features used to form a segmentation mask.
<p align="center"><img src='images/improved_unet.png'></p>

### Google Colab

This model has been developed using Google Colab. Henceforth, all of the logic (such as mounting drives) assumes that setup. There are some preliminary steps if you would like to run it within that environment.

1. You will need to upload the dataset to your Google Drive, along with (optionally) model.py. This could take acouple of hours depending on your internet speeds.
2. You will also need to update the directory strings to be the relative location of your dataset

### Running on Personal Computer

If you would like to run this on your computer then please just do not run the first cell of main.ipynb. Everything else should work accordingly (as long as you update dataset paths)

### Results

The model was trained with a _/_/_ test/train/validation split for __ epochs.


## Example output

## Loss Function Plots

### Dependencies 
- Python 3.7
- tensorflow
- sklearn
- glob
