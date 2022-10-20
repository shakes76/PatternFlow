# Generative VQVAE using PixelCNN on the ADNI Dataset

## Overview

The project implements a generative model on the ADNI dataset to generate new brain images. 
This is done by training a VQ-VAE (Vector Quantized Variational AutoEncoder) on the training images, 
and then using it to train a PixelCNN prior on encoded codebook samples. The trained PixelCNN 
then generates new codebook samples which are decoded by the VQ-VAE to generate new images.

## Models

### VQVAE

### Pixel CNN

## Data pre-processing

## Training the model


## Results


## Improvements


## Dependencies
- tensorflow >= 2.9.2
- numpy >= 1.21.6
- matplotlib >= 3.2.2
- tensorflow_probability >= 0.16.0

## References

References for understanding and creating the VQ-VAE Model
- https://keras.io/examples/generative/vq_vae/
- https://www.kaggle.com/code/ameroyer/keras-vq-vae-for-image-generation/notebook
- https://www.youtube.com/watch?v=VZFVUrYcig0

Tutorial referenced for creating the PixelCNN model
- https://keras.io/examples/generative/pixelcnn/
- https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173

Customising a tf.keras model
- https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
- https://www.tensorflow.org/guide/keras/custom_layers_and_models

Academic Paper references for VQ-VAE and PixelCNN
- https://arxiv.org/pdf/1711.00937.pdf
- https://arxiv.org/pdf/1606.05328.pdf
- https://arxiv.org/pdf/1601.06759v3.pdf


<p align="center"><img src=""></p>