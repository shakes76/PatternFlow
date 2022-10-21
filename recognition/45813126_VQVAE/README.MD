# VQ VAE on the OASIS brain dataset

## The algorithm

### Description
A Vector-Quantized Variational Autoencoder, commonly known as a VQ VAE, is a generative model which utilises vector quanitization in order to learn and comprehend patterns of data, in this case, in relation to Magnetic Resonance Imaging (MRI) of brains in the OASIS dataset. 

### Problem to solve
The VQ VAE aims to learn and understand discrete representations of the images produced by the MRI of brains. After training, the model should be able to generate accurate images of the brain, based on certain features of brains that the model found in training.

### How it works
The VQ VAE model architecture is comprised of three main parts, an encoder, VQ layer and then a decoder. The encoder and decoder both contain three convolutional layers. The decoder is the same as the encoder, except reversing its operations. The general idea is as follows. First, the encoder takes some input data and encodes it into the latent vector representation. The VQ layer then takes this vector and selects embeddings based on distance and outputs a quantized vector of the same size. This quantized vector is then passed through the decoder and the decoder attempts to recreate the original input, which in this case is an MRI of a brain.

# ADD FIGURES IN
![](../../../../UQ/2022/Sem%202,%202022/COMP3710/Report/VQVAE_architecture.png)
https://miro.medium.com/max/828/1*miNfFc9qT5PrS7ectJa_kw.png
https://medium.com/analytics-vidhya/an-overview-on-vq-vae-learning-discrete-representation-space-8b7e56cc6337


## Requirements

### Dependencies
- numpy >= 1.23.4
- tensorflow >= 2.10.0

## Reproducibility of results
- data loading, etc?

## Examples
- inputs, outputs and plots

# DESCRIBE ANY SPECIFIC PRE-PROCESSING YOU HAVE USED WITH REFERENCES IF ANY. JUSTIFY YOUR TRAINING, VALIDATION AND TESTING SPLITS OF THE DATA


