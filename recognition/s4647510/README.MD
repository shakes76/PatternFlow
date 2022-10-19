# VQVAE using OASIS brain dataset

This project is a generative model of the OASIS brain dataset using a Vector-Quantized Variational Autoencoder (VQVAE) and a PixelCNN for image generation. 

## Algorithm Description

### VQVAE

The VQVAE is a modified version of a standard Variational Autoencoder where it operates on a discrete latent space rather than a continuous distribution.

Firstly, the VQVAE involves an encoder convolutional network that creates a downsampled code for each image. The VQVAE also involves a convolutional network decoder, which transforms a code back into an image.
The latent space in this autoencoder is a trainable codebook which is quantised rather than being a continuous normal distribution.

### PixelCNN

Once the VQVAE has been trained, the PixelCNN can use the VQVAE to generate images one pixel at a time, with each next pixel value being determined by the currently generated pixels.
The PixelCNN uses two types masked convolutional layers to mask unpredicted pixels.

Mask type A is used on the first convolutional layer and only allows previously generated pixels to be seen.
Mask type B is used on all subsequent layers and allows previously generated pixels and the currently generated pixel to be seen.

The PixelCNN is trained to learn a probability distribution. Once trained, codes can be decoded by the VQVAE to generate images.

## Dependencies

## Usage
