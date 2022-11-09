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





<img src="https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/1_fN-q2XG9CTii8S6Xh8SIyg.gif?raw=true" width="256" height="256" align = "center"/>

<div style="text-align:center">
<a href="https://towardsdatascience.com/implementing-deep-convolutional-generative-adversarial-networks-dcgan-573df2b63c0d" target="_blank" >Image source.</a></div>




DC-GANs were first introduced in the research paper <a href="https://arxiv.org/abs/1511.06434" target="_blank" >Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a>
 by *Alec Radford, Luke Metz, Soumith Chintala.* Since then the GANS have been customised and developed for a variety of applications that doesn't limit to the image data. 


A successful GAN model is a one where the generated images look so real that the discriminator cannot identify a fake image from a real one. In terms of accuracy, the discriminator accuracy should reach 50% for a dataset with half real and half generated images. 

The best example to understand the power of GAN can be found on the website  https://thispersondoesnotexist.com/

The website generates a new image of a person that does not exis in real life. in the back-end, it uses a modified version of the DCGAN model called <a href=" https://arxiv.org/abs/1912.04958" target="_blank" >Style-GAN</a>  that identifies different parts of faces to train the generaotr better. The end result is shockingly real.

<img src="https://thispersondoesnotexist.com/image" width="256" height="256" />
<a href="https://thispersondoesnotexist.com/" target="_blank" ><div align="center">This person Doesnot Exist.</div></a></a>







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

<img src="https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/brain_train.PNG?raw=true" alt="drawing" width="400"/>

<a href="https://www.oasis-brains.org/" target="_blank" ><div align="center">OASIS Brain MRI Images data</div></a>




## Tools and Techniques (Libraries and dependencies for the project)

<img src="https://cdn-images-1.medium.com/freeze/max/1000/1*ZMzAVt_1ZL4AaN-hGW1-0w.png?q=20?raw=true" alt="drawing" width="400"/>

<a href="https://cdn-images-1.medium.com/freeze/max/1000/1*ZMzAVt_1ZL4AaN-hGW1-0w.png?q=20" target="_blank" >



1. <a href="https://www.tensorflow.org/" target="_blank" >**Tensorflow:**</a> The project uses Tensorflow as a base platform. TensorFlow is an end-to-end open source platform for machine learning. It uses a Tensor as a basic data structure which is  like an n-dimensional matirix. Tensor makes the data manipulations faster and effective and hence improve the overall processing of the models. For implementing Tensorflow on a GUI, Jupyter notebook is used.


2.  <a href="https://keras.io/about/" target="_blank" >**Keras:**</a> Keras is a deep learning API written in Python, Which implements Tensorflow as the platform. The key idea behind Keras was to develop a module that can enable faster implementation of machine learning and deep learning algorithms. For this project, following Keras APIs were used:

    2.1. **Models:** Model works as a wrapper for all the layers in a neural network. It creats the network as a stack of layers. The simplicity and effectiveness of the layer by layer architecture makes the model versatile and generalisable to suit any needs.
    
    2.2. **Layers:** Layers are the basic building blocks of neural networks in Keras. Each layer can be of a specific functionality. It can either be a structure representing nodes in a basic neural network or an array of nodes for Convolutio neural network. It can also be like a function implementation layer like dropouts, maxpooling, optimizers, etc. 
    
    2.3 **Dataset preprocessing:**  Keras dataset preprocessing utilities, located at tf.keras.preprocessing. These were used to preprocess data into correct structure and make it fit for use.
    
    2.4. **Optimizers:** An optimizers are the class of optimising functions to be used to train the models. The optimizers compare the current results obtained from the moedel with the train data and decide how to tune the weeights and biases in teh layers. Some of the commonly used optimizers are 'SGD' and 'Adam'. Optimizers are instantiated before passing it to model.compile().
    
    2.5. **Metrics:** A metric is a function that is used to judge the performance of your model. Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric.
    
    2.6 **Losses:** The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
    
    

    
3. <a href="https://numpy.org/" target="_blank" >**Numpy:**</a> Numpy is a Python library to implement array-like datastructures. Although most of the project use tensors as basic data blocks, some sections use numpy data processing functions as helpers.
    
4. <a href="https://opencv.org/about/" target="_blank" >**OpenCV:**</a>   OpenCV *(Open Source Computer Vision Library)* is an open source computer vision and machine learning software library. In this project, openCV is used to load, resize and preprocess the image data.

## Method to implement the algorithm

### Following process is used to create a DC-GAN for brain Image Data.

1. Load the training dataset into the environment using openCV library.

2. Resize the images from 256x256 to 128x128 (for better processing speeds).

3. Convert the images from RGB to Grayscale.

4. Convert the images to tensor imagetype. 

5. Normalise the image pixel values from (0,255) to (-1,1) for better outputs.

6. Visualise the train image data to check if the training set is good enough for generating the fake images.

7. Create a trainbatch function that takes in train data as a whole and returns a batch of train images at each function call.

8. Create a generator function based on following attributes:

    8.1 Model: keras.sequential
    
    8.2 Input layer: keras.dense layer with 16x16x256 nodes. input shape of noise shape (here, array of size 100). Activation: LeakyRelu() , BatchNormalisation = True.
    
    8.3 1st dense layer: reshaping to 16x16 2d layer with 256 nodes.
    
    8.4 Upsampling layers: Two upsampling layers converting 16x16 to 32x32 and then 128x128 2dConv layers. BatchNormalisation = True, Activation = LeakyRelu().
    
    8.5 Final (Output) layer: 128x128x1 2DConv layer. The output to be considered as the generated image from the generator model.

8. Call the generator model and observe the structure of layers. (Use model.summary())

9. Create Discriminator model based on following attributes:

    9.1 Input layer: 2dConv layer with 64 nodes and input shape (128x128x1).
    
    9.2. Downsampling layers: Two downsampling layers converting 128x128 image to 64x64 -> 32x32. Add dropout layers with activation function = LeakyRelu().
    
    9.3 Flatten Layer: Convert 2DConv layer to dense layer.
    
    9.4 Output layer: layer with 1 node representing a binary output whether the image is a fake or a real.

10. Call the discriminator model to visualise the structure of layers.

11. Define loss function as follows:

    11.1 Loss function: tf.keras.losses.BinaryCrossentropy()
    
    11.2 Generator loss: define a function to check the loss of generated image with all ones. The idea here is that we want to train generator thinking it always creates real images from input noise. This means the loss has to be compared with ones.
    
    11.3 Discriminator loss: define a function to calculate the loss of real images and fake images. The key idea here is to assume real image output as always ones and fake image outputs as always zeros. The total loss of the discriminator is defined as real loss plus fake loss. over time, this loss becomes comparable to the generator loss.
    
    
12. Optimizer functions for discriminator and generator: adam optimizer (alpha = 1e-4).

13. Define the train step function to do the following:

    13.1 Get the random normal noise of size noise dimension (here 100-array).
    
    13.2 Generate n images from generator where n is the batch size of the training loop.
    
    13.3 Use discriminator to calculate the fake output (input: generated images) and real output (input: train images).
    
    13.4 Calculate the gradients for real and fake outputs.
    
    13.5 Apply the new generated gradients to generator and discriminator
    
    13.6 Return generator loss and discriminator loss for visualisations.
    
14. Define Structural similarity function based to do the following;
    
    14.1 Get the generated image and test data as input:
    
    14.2 For each test image, calculate the ssim for generated image using tf.image.ssim().
    
    14.3 return the maximum ssim.
    
15. Define functions to handle image outputs for each epoch to do following:
    
    15.1 Get the noise as input.
    
    15.2 Generate images using the generator for the input noise.
    
    15.3 Plot the generated images.
    
    15.4 for every 5th epoch, save the generated set of images to (root)/gen_images as outputs.
    
16. Define the train function to do the following:

    16.1 Take dataset and #epochs as input.
    
    16.2 Create an iteration loop to run for #epochs.
    
    **For each loop,**
    
    16.3 Call the train step function with batch datset as input. Store generator loss and discriminator loss values into seperate lists.
    
    16.4 Call output image handler function to generate and save image.
    
    16.5. Take a random sample from generated images to test the SSIM with test dataset. Store the value into list.
    
    16.6 display the generated images for each epoch and the SSIM value.
    
17. Call the training function to run on training batch and for #epochs.

18. Observe the trends in generator and discriminator loss by plotting them.

19. Observe trends in SSIM w.r.t epochs by plotting it.

20. Create a function to take generated images as inputs and return a .GIF file to visualise the progress of the training. Call the function.


    
    
    
    
    
    
    

    
    
    

## Instructions to run the code

1. Download the helper files and the main notebook S4562394_DC_GAN_Main_notebook.ipynb.
2. Open main notebook.
3. Provide the parameters as per the instructions mentioned in the notebook.
4. Run the notebook.

## Observations

### Plot of the Generator and Discriminator loss

<img src="https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/losses_20201104_1.PNG?raw=true" alt="drawing" width="500"/>




### Plot of the SSIM

<img src="https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/SSIM_202001104.PNG?raw=true" alt="drawing" width="500"/>




### GIF of the generated images.
<img src="https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/dcgan_20201104_1.gif?raw=true" alt="drawing" width="400"/>


### Generating random image from noise

<img src="https://github.com/agrawal-s/PatternFlow/blob/topic-recognition/recognition/Shri_GAN/Screenshots/test_images_from_noise.PNG?raw=true" alt="drawing" width="500"/>



# References

* https://en.wikipedia.org/wiki/Structural_similarity
* https://arxiv.org/abs/1511.06434
* https://www.oasis-brains.org/
* https://www.tensorflow.org/tutorials/generative/dcgan
* https://www.tensorflow.org/hub/tutorials/tf_hub_generative_image_module



