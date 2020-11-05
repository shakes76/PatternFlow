# Pix2Pix - Brain MRI Segmented Image Translation Using TensorFlow.
The objective of this recognition problem is to develop a generative model on the OASIS segmented brain MRI images using conditional-GAN.

## Description of the algorithm
Pix2Pix is a conditional Generative Adversarial Network model also known as GAN, designed for image-to-image translation.
GAN architecture is comprised of a generator model which tries to generate fake or sythetic images, followed by a discriminator which on the other hand classifies image as real (from the dataset) or fake (generated from generator).
The weights of discriminator model are updated directly, however, the weights of generator model are updated via discriminator model. The two models are trained simultaneously where the generator tries to fool the discriminator and the discriminator try to classify fake images from real ones.

Few examples of Pix2Pix GAN use cases:- 

<p align="center"> ![](images/Pix2Pix_examples.PNG) </p>



## Problem this algorithm solves
This algorithm solves image-to-image tanslantion problems. Here it is genrating brain MRI images from segmented brain MRI images.

## Working of the algorithm
On running Driver.py script, Pix2Pix_algorithm.py get triggered which performs the follwing actions in sequence as mentioned: -

1. Data loading & preprocessing - Training data & validation dataset is loaded from directory where they are stored into kernel instance. Loaded data is then preprocessed by conveting pixel values from -1 to 1.
1. Generator (U-net) & Discriminator models are initialized along with respective loss functions and adam optimizers.
1. Model is trained using gradient tape. At each training interation model is trained on number of images at a time depending upon the supplied value of variable batch_count. Also after each epoch an image from validation dataset is taken and passed to through the generator to check the model performance.
1. After model is trained, testing data set is loaded, preproccesed and finally using generator brain MRI images are generated.

High level architecture of Pix2Pix GAN model with Input, generated images from generator & ground truth image is shown below:--

<p align="center">![](images/GAN_fake.png) </p>

<p align="center"> ![](images/GAN_real.png) </p>


## Results
*1st Result-*

![](images/Result1.png)

*2nd Result-*

![](images/Result2.png)


## Dataset


## Sources
1. https://www.tensorflow.org/tutorials/generative/pix2pix
1. https://arxiv.org/abs/1611.07004