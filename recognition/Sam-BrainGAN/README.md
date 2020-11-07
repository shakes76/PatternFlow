# Using a DCGAN to produce a generative model of the OASIS brain data

## Description of the Algorithm:

The Generative Adversarial Network uses two competing models - a generator and a discriminator. The generator's goal is to turn random noise into brain images the discriminator thinks are real. The discriminator's goal is to differentiate between real and fake brain images accurately.

The generator works by performing transpose convolution (a form of upsampling convolution) on the noise. The discriminator works by performing several convolutional layers followed by a dense layer to output a single value between 0 and 1; values closer to 1 indicate the discriminator believes the image is real.  These convolution models provide millions of trainable paramenters which are optimized using the Adam Optimizer.

The optimizers are optimzing in the direction which minimizes some loss functions. The Generators loss function is higher if the discriminator outputs values closer to 0 (identifies them as fake). The discriminators loss function is higher it assigns a batch of real images values close to 0 or assigns a batch of fake images values close to 1. 

## Figures:

Below is a graph of the evolution of the Generator Loss, Discriminator Loss and SSIM at each EPOCH. For this run, we used 40 EPOCHS. 

![alt text](http://url/to/img.png)

## Dependencies:

Below is a list of conda commands to set up the virtual environment for this program:

* conda install tensorflow-gpu
* conda install tqdm
* conda install matplotlib

This program was also ran and tested and ran in the spyder environment although other IDE's should also work.

## Training-testing split:

Because of the structure of the GAN, a training-testing split is not necessary. The discriminator uses a training set of images to train against the generator.
