# A tensorflow implementation of  DCGAN



## Discription 

### The model

Deep Convolution GAN (DCGAN) include two models : generator and discriminator.

The generator creates new images and tries to fool the discriminator. 

The discriminator learns how to judge fake images from real images. 

The uniqueness of DCGAN (compared with GAN) :

 	1. Use batchnormlise in both the generator and the discriminator.
 	2. Remove fully connected hidden layers for deeper architectures
 	3. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
 	4. Use LeakyReLU activation in the discriminator for all layers.

### Problem Solved

Use OASIS brain data set to build a DCGAN model. 

The generator model of the DCGAN will create a reasonably fake images and ensure the structured similarity is 0.6 





## How the model works

### The flowchart of the  process

<img src="images/Flowchat.png" alt="Flowchat" style="zoom: 67%;" />







### Details : The structure of generator

<img src="images/Generator structure.png" alt="Generator structure" style="zoom: 70%;" />







### Details : The structure of discriminator 

<img src="images/Descriminator.png" alt="Descriminator" style="zoom:60%;" />



## Dependencies required

### Requirements:

  1. Python 3.7 

  2. Tensorflow-gpu 2.1.0

  3. Keras

  4. OpenCv

  5. IPython

     

## Example outputs

<img align="left" src="images/image_at_epoch_0000.png" alt="image_at_epoch_0000" style="zoom:110%;border:10px;margin:20px"><img align="left" src="images/image_at_epoch_0050.png" alt="image_at_epoch_0050" style="zoom:110%;border:10px;margin:20px"><img align="left" src="images/image_at_epoch_0100.png" alt="image_at_epoch_0100" style="zoom:110%;border:10px;margin:20px">













<img align="left" src="images/image_at_epoch_0150.png" alt="image_at_epoch_0150" style="zoom:110%;border:10px;margin:20px"><img align="left" src="images/image_at_epoch_0200.png" alt="image_at_epoch_0200" style="zoom:110%;border:10px;margin:20px"><img align="left" src="images/image_at_epoch_1950.png" alt="image_at_epoch_1950" style="zoom:110%;border:10px;margin:20px">

















## Result

The model will be trained use 9664 images (shape 64*64)

The model will be tested by generating 32 images and compare it with 32 real images. (since the nmber of slice is 32)

Structural Similarity(SSIM) Index will be used to validate the generated images. Our purpose is to get the  SSIM is over 0.6

The plot below show how the  structure similariy changes 

<img src="images/ssim.png" alt="ssim" style="zoom:10%;" />

1. The  SSIM changes up and down around 0.6 after 50 epochs.
2. During the whole training process, the max value of ssim is 0.727

