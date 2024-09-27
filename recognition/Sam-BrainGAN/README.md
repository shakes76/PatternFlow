# Using a DCGAN to produce a generative model of the OASIS brain data

## Description of the Algorithm:

The Generative Adversarial Network uses two competing models - a generator and a discriminator. The generator's goal is to turn random noise into brain images the discriminator thinks are real. The discriminator's goal is to differentiate between real and fake brain images accurately.

The generator works by performing transpose convolution (a form of upsampling convolution) on the noise. The discriminator works by performing several convolutional layers followed by a dense layer to output a single value between 0 and 1; values closer to 1 indicate the discriminator believes the image is real.  These convolution models provide millions of trainable paramenters which are optimized using the Adam Optimizer.

The optimizers are optimzing in the direction which minimizes some loss functions. The Generators loss function is higher if the discriminator outputs values closer to 0 (identifies them as fake). The discriminators loss function is higher if it assigns a batch of real images values close to 0 or assigns a batch of fake images values close to 1. 

## Uses of the GAN:

The purpose of creating a generative model of the human brain is it can offer us a greater insight into how the human brain can be formed out of randommness, and subsequently how that randomness affects the final product. This includes potentially the source of some diseases or irregularities in the brain. 

## Figures:

Below is a graph of the evolution of the Generator Loss, Discriminator Loss and SSIM at each EPOCH. For this run, we used 40 EPOCHS. As you can see, the Generator and Discriminator Loss fluctuates quite heavily. The SSIM hovers around 0.58-0.65 fairly consistently. 

![alt text](https://github.com/samnolan4/PatternFlow/blob/topic-recognition/recognition/Sam-BrainGAN/Output_Figure.png)

## Example Outputs:

![alt text](https://github.com/samnolan4/PatternFlow/blob/topic-recognition/recognition/Sam-BrainGAN/Output_Brain1.png)

![alt text](https://github.com/samnolan4/PatternFlow/blob/topic-recognition/recognition/Sam-BrainGAN/Output_Brain2.png)

![alt text](https://github.com/samnolan4/PatternFlow/blob/topic-recognition/recognition/Sam-BrainGAN/Output_Brain3.png)

![alt text](https://github.com/samnolan4/PatternFlow/blob/topic-recognition/recognition/Sam-BrainGAN/Output_Brain4.png)

## Dependencies:

Below is a list of conda commands to set up the virtual environment for this program:

* conda install tensorflow-gpu
* conda install tqdm
* conda install matplotlib

This program was also ran and tested and ran in the spyder environment although other IDE's should also work. To run the program, simply run the DriverScript.py program.

## Training-testing split:

Because of the structure of the GAN, a training-testing split is not necessary. The discriminator uses a training set of images to train against the generator.
