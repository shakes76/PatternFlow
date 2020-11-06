Shivaji parala
===========================================================
Student no.45512746
===========================================================
Pattern Recognition
===========================================================

Deep convolutional Generative Adversive Networks
===========================================================

Generative Adversial networks or GAN's use generator framework to generate new image from noise and discriminator framework classfies the generated image into real or fake on 
comparision with original data. As trainig goes on, weight in both generator and discriminator are adjusted to optimal values to give desired output.End goal is generating images 
similar to training image data.

Deep convolutional Generative Adversive Networks or DCGAN's are most fundamental implementaion of GAN'S involving deep convolutional layers in the network to generate image and 
classify the generated image."DCGAN.s" were introduced by Alec Radford & Luke Metz in their article "Unsupervised representative learning with Deep convolutional geneartive  
adversalnetworks.

![DCGAN](https://gluon.mxnet.io/_images/dcgan.png)

Above picture display the basic architechture of Deep convolutional Generative Adversive Networks.

# The Architecture of DCGAN:

## Generator:

Generator mainly consists of deconvolutional layers or upsampling layers.Upsampling can be described as reverse of covolution and trnspose convolution layers perform this rask.
In this project, Initially noise of shape (100,) is passed through the network after few deconvolutional layers output image of size (128,128,1) is generated.

## discriminator:

This can be viewed a simple binary classifier containing many convolutional layers .finally flattened data is used to generate probability to classify the images,i.e, real or 
fake.

# Internal components of Generator and Discriminator:

## No max or average pooling:

As network only contains convolution layer we do not use max pooling for downsampling. Instead all the operations will be through strided convolutions.

## Batch normalization:

I will be using batch normalization in both generator and discriminator as it normalizes input data to each unit uin a layer and also helps in rectifying poor initialization problem.

## Activation functions:

In both generator and discriminator i will be using leakyrelu activation function after convolutional and batch normalization layers.For last convolutional layer in generator tanh function is used.


