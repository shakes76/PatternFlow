# Readme.md file for DCGAN Brain image data.

# Author: Shrikant Agrawal
## Student ID: S4562394
## Course Code: COMP3710
## Course Name: Pattern Recognition and analysis
## Assessment: Final Assignment

**Note to the readers:** This notebook provides a brief introduction to the DC-GANs and their application to generate Brain MRI images available at <a href="https://www.oasis-brains.org/" target="_blank" >https://www.oasis-brains.org/</a> 

## Project Aim:
### Create a generative model of the OASIS brain or the OAI AKOA knee data set using a DCGAN that has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6. 

## Expected Outcomes:

* Design and train a DCGAN model bsed on the research <a href="https://arxiv.org/abs/1511.06434" target="_blank" >Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a>.

* Write a module that implements the components of the model either as a function or a class that can be called by a separate driver script containing a main function. Numpy should not be used and must be written in TF. 

* Include a driver script that shows example usage of the module and runs to solve the recognition problem with appropriate plots and visualisations relevant to the problem.

# DC GAN


DC GAN or GAN: *Deep Convolutional Generative Adversarial Network* is the combination of two neural network models that are trained simultaneusly using an adversarial process.

The goal of GAN is to take a set of random input array and generate near-real images based in the input images. The generator component of the model creates the images from the input array, while the discriminator component tries to identify whether the generated image from generator is real or fake. Over the time, generator improves by taking the feedback from discriminator and modifying its weights and biases to make the generated images resemble more to the training images.




<p align="center">
![Alt Text](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/1_fN-q2XG9CTii8S6Xh8SIyg.gif?raw=true)
</p>
<div align="center">Training GAN on MNIST handwritten digits data.
<a href="https://towardsdatascience.com/implementing-deep-convolutional-generative-adversarial-networks-dcgan-573df2b63c0d" target="_blank" >Image source.</a></div>



GANs were first introduced in the research paper <a href="https://arxiv.org/abs/1511.06434" target="_blank" >Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a>
 by *Alec Radford, Luke Metz, Soumith Chintala.* Since then the GANS have been customised and developed for a variety of applications that doesn't limit to the image data. 


A successful GAN model is a one where the generated images look so real that the discriminator cannot identify a fake image from a real one. In terms of accuracy, the discriminator accuracy should reach 50% for a dataset with half real and half generated images. 

The best example to understand the power of GAN can be found on the website  https://thispersondoesnotexist.com/

The website generates a new image of a person that does not exis in real life. in the back-end, it uses a modified version of the DCGAN model called <a href=" https://arxiv.org/abs/1912.04958" target="_blank" >Style-GAN</a>  that identifies different parts of faces to train the generaotr better. The end result is shockingly real.

<img src="https://thispersondoesnotexist.com/image" width="512" height="512" />
<a href="https://thispersondoesnotexist.com/" target="_blank" ><div align="center">This person Doesnot Exist.</div></a></a>






![Test Image](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/fake_celebrities.png?raw=true)
<div align="center">These images are generated using the GAN network trained on celebrity faces dataset.</div>

## **Big Picture Process:**

1. Generator generates a random image of the size similar to the training data based on a random seed.
2. Discriminator checks if the input image from generator is real or fake.
3. Outputs from discriminator are taken as loss function for generator. Using these results from the discriminator outputs, Generator trains itself.
4. Combiner helps in the whole process by stacking up generator and discriminator as one black box. it takes noise as input => generates images => determines validity.
5. When Generator is trained, Discriminator is static. When Discriminator is trained, Generator is static.
6. Finally after training enough, the generator is able to create images that are very similar to the training data. The discriminator is unable to distinguish between the training data and the fake ones created by generator. 


![Test Image](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/dcgan_architecture.png?raw=true)
<a href="https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html" target="_blank" ><div align="center">Image: GAN Architecture</div></a>


## Dataset: 

### Brain MRI scans from <a href="https://www.oasis-brains.org/#about" target="_blank" >OASIS.org </a> 

The dataset used for the project is a collection of data for over 1000 patients while several ongoing projects by WUSTL Knight ADRC over the course of 30 years. It is an opensource dataset with data use agreement provided by the organisation. 

For training the GAN, we're using the crossectional MRI scans of patients' brains. Each brain scan has 32 cross sections. The data is stored as 256x256 pixel image stored in a *.PNG* format. An example of the image data is shown below.

![Train Image](https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/brain_train.PNG?raw=true)
<a href="https://www.oasis-brains.org/" target="_blank" ><div align="center">OASIS Brain MRI Images data</div></a>




## Tools and Techniques (Libraries and dependencies for the project)

![TF_with_keras](https://cdn-images-1.medium.com/freeze/max/1000/1*ZMzAVt_1ZL4AaN-hGW1-0w.png?q=20?raw=true)
<a href="https://cdn-images-1.medium.com/freeze/max/1000/1*ZMzAVt_1ZL4AaN-hGW1-0w.png?q=20" target="_blank" >



1. <a href="https://www.tensorflow.org/" target="_blank" >**Tensorflow:**</a> The project uses Tensorflow as a base platform. TensorFlow is an end-to-end open source platform for machine learning. The GUI is implemented using Jupyter Notebook. 


2.  <a href="https://keras.io/about/" target="_blank" >**Keras:**</a> Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result as fast as possible is key to doing good research. For this project, following Keras APIs were used:

    2.1. **Models:** Model works as a wrapper for all the layers in a neural network. It creats the network as a stack of layers. The simplicity and effectiveness of the layer by layer architecture makes the sequential model versatile and generalisable to suit any needs.
    
    2.2. **Layers:** Layers are the basic building blocks of neural networks in Keras. A layer consists of a tensor-in tensor-out computation function (the layer's call method) and some state, held in TensorFlow variables (the layer's weights).
    
    2.3 **Dataset preprocessing:**  Keras dataset preprocessing utilities, located at tf.keras.preprocessing, help you go from raw data on disk to a tf.data.Dataset object that can be used to train a model.
    
    2.4. **Optimizers:** An optimizer is one of the two arguments required for compiling a Keras model. You can either instantiate an optimizer before passing it to model.compile() , as in the above example, or you can pass it by its string identifier. In the latter case, the default parameters for the optimizer will be used.
    
    2.5. **Metrics:** A metric is a function that is used to judge the performance of your model. Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric.
    
    2.6 **Losses:** The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
    
    

3. Numpy
4. OpenCV


```python

```

* Step 1. Title For each experiment there must be a title or heading. Remember to include the date and who conducted the experiment.
* Step 2. Aim
             There must be an aim stating what this experiment intends to do or find out.
* Step 3. Hypothesis
             A prediction about what you think is going to happen.
* Step 4. A list of equipment or materials
            What you use and the quantity of each must be included.
* Step 5. Method
             Using steps and explain what happened in each step of the experiment. Remember to state how much (quantity) was added to                    what and where using which apparatus. Include a diagram at the end of the method to show how to set up the equipment or                        apparatus. This has to be detailed enough for someone else to replicate the experiment.
* Step 6. Results
             The results and observations of the experiment are recorded here, they can be (preferably) in table, list or paragraph form.
* Step 7. Discussion or Analysis
             Write down what you discovered about the experiment, including what was difficult or went wrong. The focus of your discussion                    should be based around what you think your results show about the experiment. You can include ideas for further experiments, an              explanation of problems and how to overcome them.
* Step 8. Conclusion
            A short summary of what was discovered by the experiment. This should be concise and answer the aim.

* Incredients for a gan

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
