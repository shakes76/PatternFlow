# Tensorflow Implementation of  Improved UNET

## Description 

### Google Colab

This model has been developed using Google Colab. Henceforth, all of the logic (such as mounting drives) assumes that setup. There are some preliminary steps if you would like to run it within that environment.

1. You will need to upload the ISIC dataset to your Google Drive, along with model.py. This coudl take acouple of hours depending on your internet speeds.
2. You will also need to update the directory strings to be the relative location of your dataset

### Running on Personal Computer

If you would like to run this on your computer then please just do not run the first cell of main.ipynb


## Architecture
Below is an image taken from a Berkely post (https://ml.berkeley.edu/blog/posts/vq-vae/) which shows the VQ-VAE architecture.
<p align="center"><img src='https://i.imgur.com/R9VMWD6.png'></p>