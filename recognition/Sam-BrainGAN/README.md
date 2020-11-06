# Using a DCGAN to produce a generative model of the OASIS brain data

## Description of the Algorithm:

The Generative Adversarial Network uses two competing models - a generator and a discriminator. The generator's goal is to turn random noise into brain images the discriminator thinks are real. The discriminator's goal is to differentiate between real and fake brain images accurately.

The generator works by performing transpose convolution (a form of upsampling convolution) on the noise. The discriminator works by performing several convolutional layers followed by a dense layer. These convolution models provide millions of trainable paramenters which are optimized using the Adam Optimizer.

The optimizers are optimzing in the direction of gradient loss on some loss functions. The generator's loss function evaluates   

## Dependencies:

Below is a list of conda commands to set up the virtual environment for this program:

* conda install tensorflow-gpu
* conda install tqdm
* conda install matplotlib

This program was also ran and tested and ran in the spyder environment although other IDE's should also work.

## Training-testing split:

Because of the structure of the GAN, a training-testing split is not necessary. The discriminator uses a training set of images train against the generator.