# Generator Model of the OASIS Brain data set Using StyleGAN

## Introduction

### GAN

Generative Adversarial Networks, are an effective approach for training deep convolutional neural network models for generating synthetic images.

Training a GAN model involves two models: a generator used to output synthetic images, and a discriminator model used to classify images as real or fake, which is used to train the generator model. The two models are trained together in an adversarial manner, seeking an equilibrium.

### progressive growing GAN

Progressive Growing GAN uses a generator and discriminator model with the same general structure. The key of progressive growing GAN is it starts with small  images, like 4×4 pixels.

During training, new blocks of convolutional layers are systematically added to both the generator model and the discriminator models.




