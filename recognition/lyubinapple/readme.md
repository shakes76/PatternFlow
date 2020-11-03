# **COMP3710 Report**
The report is divided into two parts. The first part is to use improved Unet to implement segmentation of OASIS brain dataset. The second part is to build the generative model of OAI AKOA knee dataset using DCGAN algorithm.

The structure of the model is presented in the appendix in the form of images (call plot _model function of tensorflow).


> **Dependency:**
- Python = 3.7
- Tensorflow = 2.1.0
- IPython
- Matplotlib
- PIL
- zipfile
- glob
>**1. Improved unet for egmentation of OASIS dataset**

**Problem Discription:**

The aim of this problem is to use improved unet to complete the segmentation of the brain MRI images. Improved Unet is a optimization of  Unet. It introduces many new modules, such as location module, context module and segmentation layer. Besides, it uses leaky ReLu instead of ReLu. One thing worth noting is that there are three segmentation layers, and their results are added to get the final output. The dimensions of these three segmentation layers are different. So first upsampling to unify the dimensions, and then add them.


**Data split:**
I use the same data split method to traditional Unet. According to the OASIS dataset. According to the OASIS data set, I transfer data from different folders to train, test and validation variables. Use train dataset and validation dataset to fit the model. Then the train loss and accuracy of each epoch, as well as the validation loss and accuracy is printed as output.

**Evaluation method:**

I write a function to cumpute the dice similarity coefficient at the end of the model. At the beginning, I took this evaluation function as a parameter of metrics. But it doesn't work (System hint: unknown metric function). So I keep using accuracy during compile step and write seperate function to evaluate at the end.

**Output and Performance:**
After 10 epochs, the dice similarity coefficient is: 0.9778, meets the requirements of 0.9. The output of dice similarity coefficient is as below. 

Dice similarity coefficient is:  tf.Tensor(0.9777626, shape=(), dtype=float32)
Besides, the val_accuracy is 0.9866. 

Example of prediction visualisation is as follow:

![Getting Started](ImprovedUnetPredict.png)

>**2. DCGAN for OAI AKOA dataset**

**Discription:**

I build DCGAN model for problem 6. DCGAN is an optimization of traditional GAN. I have already contacted the construction of the GAN model during Demo2. So this time I completed DCGAN on the basis of GAN. Generative Adversarial Networks (GANs) are a deep neural network structure consisting of two networks. It is an unsupervised learning method, including Generator and Discriminator. The generator network is a deconvolution network. It generates a random noise vector and upsamples it into a picture. The discriminator network is a standard convolutional network that can classify pictures. It is a two-classifier to mark pictures as true or false. GAN is jointly trained by the generator and the discriminator, the generator is used to generate samples to solve the maximization problem, and the discriminator is used to obtain the globality.
According to the DCGAN paper, the characteristics of my model are:
1. Use convolutional layers instead of fully connected layers
2. Use step-size convolution instead of upsampling, which improves the stability of GAN training and the quality of generated results
3. Use leakyRELU instead of RELU to prevent gradient sparseness
4. The output layer of the generator uses tanh
5. Use adam optimizer to train GAN, set the learning rate to 0.0002
6. I try to add batch normalization function into generator and discriminator. But the performance is bad. So in the end I did not use this function.

**Data split:**
I read all of images in OAI dataset. Then I use 15000 images to train the model. During training, the batch size is set to be 128. In each batch, half to it is used to generate real samples. While half to it is used to generate fake samples. Finally, form a batch of weight updates together.

**Evaluation method:**

I used the built-in SSIM function of tensorflow. It just needs to pass in the tensors of the two sets of images to be compared to get the result.

**Performance:**

After 50 epochs based on 15000 images, the prediction result is plot as follow:

![Getting Started](DCGANPredict.png)

This result is clear but the SSIM result is not good enough which is 0.11.

>**Appendix**

**Structure of Improved Unet:**

![Getting Started](ImprovedUnet.png)

**Structure of DCGAN**
1. Generator
![Getting Started](Generator.png)
2. Discriminator
![Getting Started](Discriminator.png)
3. DCGAN![Getting Started](Gan.png)