# A tensorflow implementation of  DCGAN

## Discription

Deep Convolution GAN (DCGAN) include two models : generator and discriminator.

The generator creates new images and tries to fool the discriminator. 

The discriminator learns how to judge fake images from real images. 

The uniqueness of DCGAN (compared with GAN) :

	1. Use batchnormlise in both the generator and the discriminator.
 	2. Remove fully connected hidden layers for deeper architectures
 	3. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
 	4. Use LeakyReLU activation in the discriminator for all layers.



