# Pattern Recognition: Generative Adversarial Network for MRI Generation
## Author: Erik Brand
### Dataset: OAI AKOA Knee

## Description
A Deep Convolutional Generative Adversarial Network is a generative model that is able to generate images from a latent space. It consists of two individual convolutional networks: a generator and a discriminator. These components are coupled such that they compete against each other during training; the generator attempts to generate a realistic image from the latent space, while the discriminator learns to distinguish between generated images fed from the generator and actual training data. In this particular application, the generator-discriminator network is trained on Knee MRI data such that the generator is able to create realistic 128x128 MRI images from a latent space. 

## How It Works
The discriminator network is comprised of three CONV-MaxPool-LeakyReLU blocks, followed by two fully-connected layers, while the Generator consists of five CONVTranspose-ReLU blocks. The training proceeds by first training the discriminator. A randomly batched sample of training data is fed through the discriminator and labelled with a 1 to indicate that it's real. At the same time, a batch of latent data is randomly generated from a standard normal distribution, labelled 0, and fed through the generator (which isn't trained in this iteration), before being fed through the discriminator. The total discriminator loss on both real and fake samples is calculated and back-propogated to train the discriminator. Following this, latent data is fed through the generator and discriminator, but this time is labelled with a 1 in an attempt to trick the discriminator into thinking the fake images are real. This time, the discriminator's weights are held constant while the loss is back-propogated through the generator. This process continues for every batch of data, for the specified number of epochs.

## To Run
There are three files required to run this algorithm:
* GAN.py - Defines the DCGAN and training function
* DataUtils.py - Defines auxiliary functions for loading and visualising data
* Driver.py - Driver script for running the training function

The file Driver.py demonstrates how to run the algorithm. To run the algorithm, specify the required hyperparameters, then call the train() method: `train(data_path, output_path, epochs, batch_size, latent_dim, generator_input_dim, learning_rate_generator, learning_rate_discriminator, debug)` 

Parameter | Description
--------- | -----------
data_path | The filepath to the training data
output_path | The filepath to the desired output location
epochs | The number of iterations of the entire dataset
batch_size | The size of each training batch
latent_dim | The size of the latent dimension
generator_input_dim | The side length of the square input size for the generator
learning_rate_generator | The learning rate of the generator
learning_rate_discriminator | The learning rate of the discriminator
debug | Whether to output training images, generated images, saved models after each epoch for debugging

The generator and discriminator loss will be output after each epoch. Once training is complete, a plot of the loss over the training process is prodcued (example_output.png), along with some example images generated from the latent space with the final version of the generator (loss_plot.png), and finally the SSIM score of the generated images compared to some test images is recorded in the file SSIM.txt. If debug = True, models, example generated images, and a small sample of the training images are saved after each epoch in the /Resources folder.

NOTE: To use debug = True, please create the following folders in the /Resources directory:
* TrainImages
* Intermediate
* Models


## Dependencies
The algorithm itself sits within a self-contained module (GAN.py). This does not depend on any external libraries except for TensorFlow. However, there are a number of auxiliary methods for loading and visualising data that have the follwoing dependencies:
* TensorFlow (>=2.1.0)
* pydot (>=1.4.1)
* glob
* sys
* numpy (>=1.19.1)
* matplotlib (>=3.3.1)


## Example Output
![Example Output](Resources/example_output.png)  
If the above image is not visible, please navigate to ./Resources/example_output.png


## Example Loss Plot
![Example Loss Plot](Resources/loss_plot.png)  
If the above image is not visible, please navigate to ./Resources/loss_plot.png


## Data Split
GANs are unsupervised machine learning algorithms, and as such the entire dataset is usually used for training the model. This is because the performance is often measured by generating images from the latent space and assessing the results visually. In this case, the random values generated from the latent space are essentially the 'test' dataset. However, as this algorithm assesses the model based on the SSIM, a subset of the data was reserved for validation and testing. All of the images from a single patient were separated from the training set to create each of the validation and test subsets. Each one of these validation/testing images was compared to a sample of generated images to calculate the average SSIM over the validation and testing sets. It was decided that only a single patient would be reserved for each of the validation and testing set as SSIM is only a secondary measure of performance, whereas the visual inspection is the primary measure and does not require a validation/test subset of actual data.


### A Note on SSIM
SSIM is a measure of image similarity and quality. It is used to compare a 'reference' image of perfect quality to a version of the *same* image that has a lower quality. The original SSIM is defined only on grayscale images, and as such the data loading functions defined in DataUtils.py load the images as grayscale. SSIM probably isn't the best measure for this particular application. As can be seen on this website (https://www.cns.nyu.edu/~lcv/ssim/), a blurred version of the *same* image returns a SSIM of 0.69, while here (https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html) the *same* image with added noise returns a SSIM of 0.15. In this particular application, we are generating entirely new MRI images from the latent space. This means that even if the generated images are of high quality, they will have a different bone structure to any of the training/testing images. This means that the base image is fundamentally different, hence the quality cannot be accurately compared with SSIM. To test this, the SSIM was calculated across two real images from two different patients, which revealed they had a SSIM of 0.13. This algorithm returns a SSIM of roughly 0.17, which I believe is a resonable result for this application.

