# Deep Convolutional Generative Adversial Network(DCGAN) for OASIS Brain data-set

### Description

The main idea of GAN is to generate the new random images(fake images), which looks more realsitic to the input images. It consists of Generator - to generate fake images, Discriminator - to classify the image as fake or real, and adversarial network that pits them against each other. 

DCGAN is an extension to the GAN architecture with the use of convolutional neural networks in both generator and discriminator part. Implenentation of the DCGAN model here, follows the the paper [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) by Radford et. al. It suggests the constraints on the model required to effectively develop the high-quality generator model.

### Problem

Generating new brain images can be seen as a random variable generation problem, a probabilistic experiment. The sample image from the input OASIS brain dataset is shown below. Its 256 X 256 size with 1 grey-scale channel. Each image is a vector of 65,536-dimensions. We build a space with 65,536 axes, where each image will be a point in this space. Having a probability distribution function, maps the each input brain images to the non-negative real number and sums to 1. GAN generates the new brain image by generating the new vector following this probability distribution over the 65,536-dimensional vector space, which is very complex one and we dont know how to generate this complex random variables. The idea here is an transform method, generating 65,536 uniform random variables(as a noise vector) and apply the complex function to this variable, where this complex functions are approximated by the CNN in the generator part and produce the 65,536-dimensional random variable that follow the input brain images probability distribution.

