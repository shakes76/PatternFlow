# Variational Autoencoder


## Overview
To generate brain images, the algorithm I applied is Variational Autoencoder. 
In the scenario of computer vision, an autoencoder is trained to learn the latent 
representation (or a single point) of an image in a reduced dimension. 
However, rather than using a fixed vector to compress images, a Variational Autoencoder 
learns to compress the images into a range of possible values (i.e., a probability distribution). 
Then we sample vectors (points) from the latent distribution produced by the 
encoder and feed them to the decoder to generate brain images. 
A Gaussian distribution is used here as the latent distribution.


## Dependencies
The dependencies are also specified in requirements.txt.

### Tensorflow: 
To install Tensorflow, use the following command:
```
pip install tensorflow
```

### matplotlib
To install matplotlib, use the following command: 
```
pip install matplotlib
```


## Dataset Preparation
Since VAE is an unsupervised learning technique, we do not need too many test images. 
In my implementation, the training set, test set, validation set contain 9664, 544, 1120 images respectively.
The model fits on the training set. The test set is used to evaluate the model performance 
by Structural Similarity Index Measure (SSIM) and Evidence Lower Bound (ELBO). 
The training set should contain more images than validation and test sets. 
The validation set is used to fine tune the hyperparameters, e.g., stride, kernel size.
To get a better estimate of model performance when fine tuning the hyperparameters, 
the validation set is larger than the test set. 
To reduce computation time in the fine tuning process, the validation set should be a lot smaller than the training set. 



## Architecture
The encoder and decoder have three layers because deep architectures can yield better compression result compared to shallow ones.
After fine tuning, the encoder has 16, 32, 64 filters for each layer.
The decoder has 64, 32, 16 filters each layer.
The stride size and kernel size are 2 and 3, respectively.
Batch normalisation layers are used to because it has been found that they can (slightly) improve the SSIM and speed up the training process.
The architecture of the VAE is visualised below. 
<p align="center">
    <img src='images/Architecture.JPG' width=80%>
</p>


## Training
In each epoch, the model is trained on the same 9664 images. 
The generated images are evaluated by SSIM.
The SSIM and ELBO after the first epoch are 0.4773 and 17951.17. 
Let’s plot four of the images generated in the first epoch below.
<p align="center">
    <img src='images/first_epoch.png' width=40%>
</p>
After 30 epochs, the SSIM and loss are 0.6467 and 17051.21. Let’s plot four of the generated images：
<p align="center">
    <img src='images/30th epoch.png' width=40%>
</p>
The ELBO and SSIM over the 30 epochs are shown below. 
<p align="center">
    <img src='images/evaluation.png' width=80%>
</p>


## Running Code
To run the project, execute `driver.py` located in `Patternflow/recognition/VAE/` 
using the following command:
```
python driver.py arg1 arg2
```
where arg1 is the directory stores the training images and 
arg2 is the directory stores the testing images.
Below is an example: 
```
python driver.py keras_png_slices_data/train keras_png_slices_data/test
```
where `keras_png_slices_data/train` stores the training images and 
`keras_png_slices_data/test` stores the test images.


## Author
Yunke QU. Student ID: 44631604.
 