# Readme for DCGAN for Brain Image Data

# Author: Shrikant Agrawal

## Course: Pattern Recognition.


```python

```

# GAN: Generative Adverserial Networks


Big Picture Process:

1. Generator generates a random image of the size similar to the training data based on a random seed.
2. Discriminator checks if the input image from generator is real or fake.
3. Outputs from discriminator are taken as loss function for generator. Using these results from the discriminator outputs, Generator trains itself.
4. Combiner helps in the whole process by stacking up generator and discriminator as one black box. it takes noise as input => generates images => determines validity.
5. When Generator is trained, Discriminator is static. When Discriminator is trained, Generator is static.
6. Finally after training enough, the generator is able to create images that are very similar to the training data. The discriminator is unable to distinguish between the training data and the fake ones created by generator. 

DC GAN or GAN: Deep Convolutional Generative Adversarial Network are the combination of two neural network models that are trained simultaneusly using an adversarial process.

The goal of GAN is to take a set of random input array and generate near-real images based in the input images. The generator component of the model creates the images from the input array, while the discriminator component tries to identify whether the generated image from generator is real or fake. Over the time, generator improves by taking the feedback from discriminator and modifying its weights and biases to make the generated images resemble more to the training images.

A successful GAN model is a one where the generated images look so real that the discriminator cannot identify a fake image from a real one. In terms of accuracy, the discriminator accuracy should reach 50% for a dataset with half real and half generated images. 

The best example to understand the power of GAN can be found on the website  https://thispersondoesnotexist.com/

The website generates a new image of a person that does not exis in real life. in the back-end, it uses a modified version of the DCGAN model called Style-GAN that identifies different parts of faces to train the generaotr better. The end result is shockingly real.




```python

```

<img src="https://thispersondoesnotexist.com/image" width="512" height="512" />

<p style="text-align: center;">This person Doesnot Exist.</p>

https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/fake_celebrities.png

![Test Image](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/fake_celebrities.png?raw=true)


```python

```


```python

```
