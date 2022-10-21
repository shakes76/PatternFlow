# VQ VAE on the OASIS brain dataset

## The algorithm

### Description
A Vector-Quantized Variational Autoencoder, commonly known as a VQ VAE, is a generative model which utilises vector quanitization in order to learn and comprehend patterns of data, in this case, in relation to Magnetic Resonance Imaging (MRI) of brains in the OASIS dataset. 

### Problem to solve
The VQ VAE aims to learn and understand discrete representations of the images produced by the MRI of brains. After training, the model should be able to generate accurate images of the brain, based on certain features of brains that the model found in training.

### How it works
The VQ VAE model architecture is comprised of three main parts, an encoder, VQ layer and then a decoder. The encoder and decoder both contain three convolutional layers. The decoder is the same as the encoder, except reversing its operations. The general idea is as follows. First, the encoder takes some input data and encodes it into the latent vector representation. The VQ layer then takes this vector and selects embeddings based on distance and outputs a quantized vector of the same size. This quantized vector is then passed through the decoder and the decoder attempts to recreate the original input, which in this case is an MRI of a brain.

# ADD FIGURES IN
![Architecture](https://miro.medium.com/max/828/1*miNfFc9qT5PrS7ectJa_kw.png)


## Requirements

### Dependencies
The following are required, with their respective versions mentioned. In some cases, a different version may be suitable; however, these have not been tested so are not encouraged.
- `python` = 3.10.6
- `numpy` = 1.23.4
- `tensorflow` = 2.10.0
- `matplotlib` = 3.5.2

## Pre-processing of data
In dataset.py, lines 16-22 (inclusive) contain the paths to the relevant data. Line 16 will need to be modified by the user who runs the program to ensure it is pointing to the correct directory on their local machine. Steps to take:
1. Download the OASIS data from https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download
2. Extract the data to a chosen location
3. Copy the path to the keras_png_slices_data folder and enter that as the value of the `path` variable on line 16.


## Reproducibility of results
- data loading, etc?

## Examples
In order to train the model, 

# JUSTIFY YOUR TRAINING, VALIDATION AND TESTING SPLITS OF THE DATA


## References
Image - https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337
