# Vector Quantized Variational Autoencoders (VQ-VAE) on OASIS brain data set.

Create a generative model of one of the OASIS brain that has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6.


## Features

- Training Loss Plots
- Example Encoded, Latent and Decoded Representations
- Samples of Generated Images


## Vector Quantised AutoEncoders:

A vector quantised autocencoder (VQVAE) is an autoencoder that uses a discrete latent variables instead of a prior distribution that is featured in a normal encoder (VAE). 

The discrete latent variables are often fit in a structure called a latent space, which is the 
Vector Quantised layer of the VQVAE. A dictionary of embeddings called a "codebook" is used to keep track of the discrete representations.

The Vector Quantised Layer works like this:

1. The output of the encoder is passed onto the the latent space.
2. The euclidean distances between the encoder output vectors and the codebook vectors are calculated.
3. The codebook vector with the smallest euclidean diustance is chosen and fed to the decoder.
4. The codebook vector is fedback to the latent space for backpropagation.

![alt text](https://github.com/Quentin1168/PatternFlow/blob/c8c10c555df913b001c0450e32ebfb5fb8ee242b/recognition/46425254-VQVAE/readme_images/Screenshot%202022-10-20%20160250.png?raw=true)

# Implementation:
## Dataset
A custom dataset file was made. The dataset that is needed is taken from https://www.oasis-brains.org/ (OASIS-1) . However, the dataset that is downloaded from that website still needs to be manually restructured in order for it to be able to be processed by the dataset file.

The dataset should be structured in the following format:
- Given the dataset, which contains the following 3 folders:
-- keras_png_slices_train
-- keras_png_slices_test
-- keras_png_slices_validation
Create a new folder for each of the folders above, for example:
-- create a folder called train and put keras_png_slices_train inside of it.
- To specify the path of the folder from now on, only go up to the folder that wraps around the original folders above.
-- For the example above, only call, for example: path1/path2/path3/train when the file path is a parameter.

These are some of the samples from the dataset:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/98928ee91bfcab16a7725e246540020f6ae678d0/recognition/46425254-VQVAE/readme_images/case_001_slice_27.nii.png?raw=true) 
![alt text](https://github.com/Quentin1168/PatternFlow/blob/98928ee91bfcab16a7725e246540020f6ae678d0/recognition/46425254-VQVAE/readme_images/case_001_slice_19.nii.png?raw=true) 
![alt text](https://github.com/Quentin1168/PatternFlow/blob/98928ee91bfcab16a7725e246540020f6ae678d0/recognition/46425254-VQVAE/readme_images/case_001_slice_11.nii.png?raw=true)
## Model Architecture:
The model architecture for the VQVAE, stored in the modules.py file is as follows:
The Latent Space specified was 16.
### Encoder:
- Conv2d, Input Filters: 3, Output Filters: Latent space, Kernel Size = 4, Stride = 2, Padding = 1
- BatchNormalisation Layer, Input Filters: Latent Space
- Activation Layer = Leaky ReLU.
- Conv2d, Input dim: latent space, Output Dim = 2 * latent space, Kernel Size = 4, Stride = 2, Padding = 1
- BatchNormalisation Layer, Input Filters: 2* Latent Space
- Activation Layer = Leaky ReLU.
- Conv2d, Input dim: 2* latent space, Output Dim = latent space, Kernel Size = 3, Stride = 1, Padding = 1
- Activation Layer = Sigmoid

### Decoder:
- ConvTranspose2d, Input Filters: Latent space, Output Filters: 2 * Latent space, Kernel Size = 4, Stride = 2, Padding = 1
- BatchNormalisation Layer, Input Filters: 2* Latent Space
- Activation Layer = Leaky ReLU.
- ConvTranspose2d, Input Filters: 2 * Latent space, Output Filters: Latent space, Kernel Size = 4, Stride = 2, Padding = 1
- BatchNormalisation Layer, Input Filters: Latent Space
- Activation Layer = Leaky ReLU.
- ConvTranspose2d, Input Filters: Latent space, Output Filters: 3, Kernel Size = 3, Stride = 1, Padding = 1
- Activation Layer = Tanh.

### VQVAE Training:
The model was trained for 15 epochs, with a learning rate of 0.001 using the adam optimiser. The loss is specified from the VQVAE loss formula from the paper.
The testing set was used to test SSIM of decoded images with their original counterparts, using the training set would be considered an overfit since that was used to train the encoding/decoding process.

Here are the results after training for 15 epochs:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/e6a45175f40c8ad5360c5cda3767f22dd68c31c5/recognition/46425254-VQVAE/readme_images/Lat_Enc2.png?raw=true)
![alt text](https://github.com/Quentin1168/PatternFlow/blob/e6a45175f40c8ad5360c5cda3767f22dd68c31c5/recognition/46425254-VQVAE/readme_images/Dec_Enc1.png?raw=true) 

The SSIM metrics calculated after running the test dataset through the VQVAE is recorded below:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/e6a45175f40c8ad5360c5cda3767f22dd68c31c5/recognition/46425254-VQVAE/readme_images/SSIM_1.png?raw=true)

The loss graph for training is recorded below:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/e6a45175f40c8ad5360c5cda3767f22dd68c31c5/recognition/46425254-VQVAE/readme_images/VQVAE_Loss.png?raw=true)

As seen with the decoded image above, it still is not very clear compared to the encoded image. However, it was only trained with 15 epochs. The model, if given more time to train would have produced decoded images with far more high quality.

### Image Generation:
Even with a trained decoder, it will not be able to generate images from scratch. A more powerful prior is needed, which can be produced with the use of a PixelCNN.

#### PixelCNN:
The PixelCNN is a model that generates models pixel by pixel, with the current pixel being generated being based on the previously generated pixels. This rule is strict, as the model does not allow the current pixel being generated to know of the future, yet to be generated pixels.

![alt text](https://github.com/Quentin1168/PatternFlow/blob/b5351924ebb95fd94b04b89a539ceca85c9ecb23/recognition/46425254-VQVAE/readme_images/pixelCNN.png?raw=true)![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/pixelCNNB.png?raw=true)

Above is an example of  pixelCNN kernel masks, there are two types of masks, A mask and B mask shown respectively. 
For a mask with the current pixel being in the dead center of the layer:
- If the mask type is A, then the current pixel is not included
- If the mask type is B, then the current pixel is included.

### PixelCNN Architecture:
The number of latent vectors used is 64.

#### Residual Block Architecture:
- Conv2d, Input Filters: # of latent vectors, Output Filters = # of latent vectors, Kernel size = 1
- Activation Layer = ReLU
- MaskedConv2d, Mask Type = B, Input Filters = # of latent vectors, Output Filters = # of latent vectors/2, Kernel Size = 3, padding = Same
- Activation Layer = ReLU
- Conv2d, Input Filters: # of latent vectors/2, Output Filters = # of latent vectors, Kernel size = 1
- Activation Layer = ReLU

The PixelCNN Architecture is specified below:

- MaskedConv2d, Mask Type = A, Input Filters = # of latent vectors, Output Filters = 4 * # of latent vectors, Kernel Size = 9, padding = Same
- Activation Layer = ReLU
- Resblock, Input Filters = 4* # of latent vectors
- Resblock, Input Filters = 4* # of latent vectors
- MaskedConv2d, Mask Type = B, Input Filters = 4 * # of latent vectors, Output Filters = 4 * # of latent vectors, Kernel Size = 1, padding = Same
- Activation Layer = ReLU
- MaskedConv2d, Mask Type = B, Input Filters = 4 * # of latent vectors, Output Filters = 4 * # of latent vectors, Kernel Size = 1, padding = Same
- Activation Layer = ReLU
- MaskedConv2d, Mask Type = B, Input Filters = 4 * # of latent vectors, Output Filters = # of latent vectors, Kernel Size = 1, padding = Valid

### PixelCNN Training:
The model was trained in many different configurations, and even more number of epochs. The most common learning rate used was 0.0003 using the adam optimiser. Cross Entropy Loss was used.
Since this is a generative model, testing and validation sets were considered unneccesary, since the model is generative in nature, and only a "reasonably clear image" is needed.

Initially, the problem ran into many problems with generation. Completely blank images were generated for the duration of training. This was later solved when it was found that a Categorical Distribution Sampler was needed to sample each of the pixels during image generation.

However, another problem arose, as images generated, even after hundreds of epochs, had the patterns and features of the data, but hardly their shape.

![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_256_Epcoh%20150.png?raw=true)

This generation was at epoch 150.

![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_256_Epcoh%20170.png?raw=true)

This generation was at epcoh 170. Features such as the carapace and the brain innards are clearly visible, but the shape was not defined.

However, after much deliberation, it was found that a smaller dimension of the original data could help with speeding up the generation training process. The model is clearly working, with features clearly being generated, however, the problem space for the generation for simply too large and the model could not generate a proper shape given all the possible likelihoods.

Therefore, all of the data, when loaded in the dataset file, was resized to be (128,128), half the size. And after 300 epochs of training, these are some samples of generated images:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent1.png?raw=true) 
![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent2.png?raw=true)
![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent4.png?raw=true) 
![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent3.png?raw=true)

These were the generated samples during training:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent%20epoch%2020.png?raw=true) ![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent%20epoch%20100.png?raw=true)

This is around epoch 20 and 100 respectively.

![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent%20epoch%20160.png?raw=true) ![alt text](https://github.com/Quentin1168/PatternFlow/blob/fe95be39b6c6e04fa1a581f9a0ca6b4fcf723fe3/recognition/46425254-VQVAE/readme_images/PCNN_Latent%20epoch%20200.png?raw=true)

This is around epoch 160 and 200 respectively.

The loss during the training process is shown here:

![alt text](https://github.com/Quentin1168/PatternFlow/blob/28903c8ff707df09fcb1dcd3a7d86145f8ec0d9f/recognition/46425254-VQVAE/readme_images/PixelCNN_Loss.png?raw=true)

### Instructions to use:
1. Open train.py, and following the documentation, run the VQVAE trainer first, and then the test to see the SSIM of the decoded images, specify a save file so that the VQVAE model can be saved.
2. Using the path to the VQVAE model, train a pixel CNN. Images should show up as plots as the model is trained. 
3. When that is done, go to predict.py and use gen_image to generate images, and VQVAE_Predict to see the visual comparison between original and decoded images, as well as their latent representations.
### Dependencies:
- Pytorch -> torch, torchvision, usage of a graphics card is highly recommended for reasonable training times.
- Matplotlib.pyplot for plotting the images and generations.
- numpy for utility
- skimage.metrics for SSIM measurement

### References:
- https://www.oasis-brains.org/ -> Dataset Used (Use OASIS one)
- https://arxiv.org/abs/1711.00937 -> VQVAE paper
- https://arxiv.org/abs/1606.05328 -> PixelCNN paper
