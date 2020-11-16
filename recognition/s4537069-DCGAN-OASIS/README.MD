# Generative DCGAN Model for OASIS Brain Dataset using TF
This model generates brain scans using the OASIS Brain dataset through the use of a Direct Convolutional Generative Adversarial Network (DCGAN). The model is fairly successful, evident using a Structure Similarity (SSIM) comparison. The SSIM upon comparing a batch of generated images via this model to a batch of real images was approximately 0.67, where the closer the SSIM value to 1 the better. This model utilised 50 unique batches from the OASIS Brain training set. 

Please refer to the bottom of this README file for dependencies, how to run the model, and further information regarding running this own your own system.

## Preprocessing Data for the DCGAN
The OASIS brain dataset is a set of preprocessed images, of which 9664 of these images are considered for training and 544 for testing. This model utilised 50 randomly chosen batches of size 16, thus each batch having 16 images. Each image was resized to a resolution of 256x256, and then had their colour pallete changed to grayscale. They were then normalised to (-1, 1) for training within the DCGAN.

## An Overview of GANs and DCGANs

### What is a GAN?
Generative Adversarial Networks (GANs) are an approach for generative modelling using deep learning methods. It is an unsupervised learning task that discovers and learns regularities and patterns in data such that it can generate new examples of those learnt patterns. GANs are a revolutionary approach to training a generative model. GANs consist of two parts, a generator model; which generates new examples via training data, and a discriminator model; which tries to classify the generator's outputted examples as real or fake. The pair are trained together, at the same time, resulting in an adversarial sort of training where the generator is trying to trick the discriminator. The final goal of the model is that the discriminator has to randomly check the generator's output as it cannot distignuish between real and fake, at which it can be said the generator is creating realistic and feasible examples.

### What is a DCGAN?
The DCGAN created by Radford et. al. in the article, "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks is an adaptation of the original GAN, using convolutional layers in both the discriminator and generator models.

## Model Summary and Explanation
Below is an explanation an of the DCGAN used in this repository to train the OASIS Brain Dataset. It explains each sub-model and the layers used to create them. A summary of the execution of training can also be found below.

### Generator
The generator starts off with a Dense layer that is fed a vector of random noise. This is then upsampled to a single 256 by 256 image. This upsampling is achieved via Conv2DTranspose layers which is a strided convolution transpose layer, followed by a batch normalisation layer to create stability via rescaling and recentering. For each upsample stage we use LeakyReLU activation. Although being another strided convolution tranpose layer, the final layer of the generator uses a tanh activation to guarantee the output to be normalised. Strides of (2,2) and kernel sizes of (3,3) were consistently used in the generator. Below is a screenshot of the model.summary() which outlines the order of the layers used in the generator and has some other minor details.

![Gen Summary](Generator_Summary.PNG)

### Discriminator
The discriminator downsamples the provided input (image) until a single scalar value is reached and then outputted. This is achieved via Conv2D layers, also known as convolution layers. This discriminator uses three convolution layers, each with LeakyReLU activation and dropout layers. A strided convolution is executed downsampling the image until the last layer. This last layer is a dense layer which outputs the single sclar value aforementioned. This value correlates to the probability of the input being real or fake. Below is a screenshot of the model.summary() which outlines the order of the layers used in the discriminator and has some other minor details.

![Disc Summary](Discriminator_Summary.PNG)


### Execution of Training

## Evaluation of Model

## Final Results

## Running the Model Locally
