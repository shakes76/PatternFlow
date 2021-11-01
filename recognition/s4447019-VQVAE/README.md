# Implementation of VQVAE for COMP3710 Project

## Algorithm description
This is the implmentation of the vector quantised variational autoencoder also using PixelCNN. This is algorithm is used to generate OASIS brain images. Figure 1 displays the model, sourced from the original paper (Neural Discrete Representation Learning,  2018).

![Figure 1. Diagram of model from original paper](/recognition/s4447019-VQVAE/images/vqvaemodel.PNG)

## Dependencies
The list of dependencies required for this implementation are as follows:

* python 3
* tensorflow 2.6.0
* tensorflow-probability 0.14.0
* numpy 1.21.3
* matplotlib 3.4.3
* scikit-image 0.18.1

## Example outputs & plots
Figure 2 shows the original and reconstructed results of 10 images.  
![Figure 2. Original and Reconstructed results](/recognition/s4447019-VQVAE/images/results1.PNG)

The Structured Similarity (SSIM) ranged between ~0.64 and 0.82. Figure 3 and Figure 4. show screenshots of the SSIM from Jupyter.  
![Figure 3. SSIM ~0.63](/recognition/s4447019-VQVAE/images/ssim1.png)
![Figure 4. SSIM ~0.82](/recognition/s4447019-VQVAE/images/ssim2.png)

Figure 5 shows the original and code results of 10 images.  
![Figure 5. Original and Code results](/recognition/s4447019-VQVAE/images/results2.PNG)

Finally, Figure 6 shows what was supposed to be the code and and generated sample images, however there was an error in implmenting this part of the network, so the images did not turn out as expected. Thought I'd just display them anyway in the hope of getting some feedback on how this part of the agorithm should have been implemented.  
![Figure 5. Code and Generated Smaples](/recognition/s4447019-VQVAE/images/results3.PNG)

## Justification of training, validation and testing split of data
The spilt between the datasets was 90% for training, 10% for validation and 5% for testing, though the validation set didn't end up being rquired. Due to the large number of usable images (total of 11,328), testing only requires a 5% share of the total images. It makes for a more robust model to use as many images as possible for the training.
