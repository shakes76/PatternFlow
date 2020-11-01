# Readme.md file for DCGAN Brain image data.

# Author: Shrikant Agrawal
## Student ID: S4562394
## Course Code: COMP3710
## Course Name: Pattern Recognition and analysis
## Assessment: Final Assignment

**Note to the readers:** This notebook provides a brief introduction to the DC-GANs and their application to generate Brain MRI images available at <a href="https://www.oasis-brains.org/" target="_blank" >https://www.oasis-brains.org/</a> 

![Alt Text](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/1_fN-q2XG9CTii8S6Xh8SIyg.gif?raw=true)<p style="text-align: center;">Training GAN on MNIST handwritten digits data.</p>
<a href="https://towardsdatascience.com/implementing-deep-convolutional-generative-adversarial-networks-dcgan-573df2b63c0d" target="_blank" ><p style="text-align: center;">Image source.</p></a>



# DC GAN


DC GAN or GAN: *Deep Convolutional Generative Adversarial Network* is the combination of two neural network models that are trained simultaneusly using an adversarial process.

The goal of GAN is to take a set of random input array and generate near-real images based in the input images. The generator component of the model creates the images from the input array, while the discriminator component tries to identify whether the generated image from generator is real or fake. Over the time, generator improves by taking the feedback from discriminator and modifying its weights and biases to make the generated images resemble more to the training images.

GANs were first introduced in the research paper <a href="https://arxiv.org/abs/1511.06434" target="_blank" >Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a>
 by *Alec Radford, Luke Metz, Soumith Chintala.* Since then the GANS have been customised and developed for a variety of applications that doesn't limit to the image data. 


A successful GAN model is a one where the generated images look so real that the discriminator cannot identify a fake image from a real one. In terms of accuracy, the discriminator accuracy should reach 50% for a dataset with half real and half generated images. 

The best example to understand the power of GAN can be found on the website  https://thispersondoesnotexist.com/

The website generates a new image of a person that does not exis in real life. in the back-end, it uses a modified version of the DCGAN model called <a href=" https://arxiv.org/abs/1912.04958" target="_blank" >Style-GAN</a>  that identifies different parts of faces to train the generaotr better. The end result is shockingly real.

<img src="https://thispersondoesnotexist.com/image" width="512" height="512" />
<a href="https://thispersondoesnotexist.com/" target="_blank" ><p style="text-align: center;">This person Doesnot Exist.</p></a>






![Test Image](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/fake_celebrities.png?raw=true)
<p style="text-align: center;">These images are generated using the GAN network trained on celebrity faces dataset.</p>



**Big Picture Process:**

1. Generator generates a random image of the size similar to the training data based on a random seed.
2. Discriminator checks if the input image from generator is real or fake.
3. Outputs from discriminator are taken as loss function for generator. Using these results from the discriminator outputs, Generator trains itself.
4. Combiner helps in the whole process by stacking up generator and discriminator as one black box. it takes noise as input => generates images => determines validity.
5. When Generator is trained, Discriminator is static. When Discriminator is trained, Generator is static.
6. Finally after training enough, the generator is able to create images that are very similar to the training data. The discriminator is unable to distinguish between the training data and the fake ones created by generator. 



![Test Image](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/dcgan_architecture.png?raw=true)
<a href="https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html" target="_blank" ><p style="text-align: center;">Image: GAN Architecture</p></a>


## The project: GAN for brain Image data


```python

```

Incredients for a gan

improving the model

producing the results

Testing the model: SSIM

The project pipeline

* dataset
* helpers/ dependencies
* generator
* discriminator
* training
* converting to images and then to gifs
* testing model accuracies and loss
* plots of accuracy, loss, ssim w.r.t epochs

Final gifs

conclusions

future work


references



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
