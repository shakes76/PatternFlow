# Generative DCGAN Model for OASIS Brain Dataset using TF
This model generates brain scans using the OASIS Brain dataset through the use of a Direct Convolutional Generative Adversarial Network (DCGAN). The model is fairly successful, evident using a Structure Similarity (SSIM) comparison. The SSIM upon comparing a batch of generated images via this model to a batch of real images was approximately 0.67, where the closer the SSIM value to 1 the better. This model utilised 50 unique batches from the OASIS Brain training set. 

Please refer to the bottom of this README file for dependencies, how to run the model, and further information regarding running this own your own system.

## Preprocessing Data for the DCGAN
The OASIS brain dataset is a set of preprocessed images, of which 9664 of these images are considered for training and 544 for testing. This model utilised 50 randomly chosen batches of size 16, thus each batch having 16 images. Each image was resized to a resolution of 256x256, and then had their colour pallete changed to grayscale. They were then normalised to (-1, 1) for training within the DCGAN.

## An Overview of GANs and DCGANs

### What is a GAN?
Generative Adversarial Networks (GANs) are an approach for generative modelling using deep learning methods. It is an unsupervised learning task that discovers and learns regularities and patterns in data such that it can generate new examples of those learnt patterns. GANs are a revolutionary approach to training a generative model. GANs consist of two parts, a generator model; which generates new examples via training data, and a discriminator model; which tries to classify the generator's outputted examples as real or fake. The pair are trained together, at the same time, resulting in an adversarial sort of training where the generator is trying to trick the discriminator. The final goal of the model is that the discriminator has to randomly check the generator's output as it cannot distignuish between real and fake, at which it can be said the generator is creating realistic and feasible examples.

### What is a DCGAN?
The DCGAN created by Radford et. al. in the article, "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks is an adaptation of the original GAN, using convolutional layers in both the discriminator and generator models.

## Model Summary and Explanation

### Generator

### Discriminator

### Execution of Training

## Evaluation of Model

## Final Results

## Running the model locally
