# Vector Quantized Variational Autoencoder (VQVAE) using OASIS dataset of brain scans
A generative model of the OASIS brain dataset using VQVAE to reconstruct a "reasonably clear images" with Structured Similarity (SSIM) of over 0.6. Examples of reconstructed images and a graph of loss rates are provided.

## Algorithm 
### VQVAE model
The VQVAE is a variational autoencoder that operates on a discrete latent space as opposed to a continious latent space in the regular VAEs. The advantage of this is simplifying the optimization. The VQVAE does this by using the idea of a discrete "codebook" that takes the prior continuous embeddings (sampled from uniform distribution) and encoded outputs and finds their discrete difference, which is subsequently put through the decoder and trained for image reconstruction. The encoder is a convolutional network using downsampling, whereas the decoder uses upsampling in its convolutional network. 

## Implementation 
### Dataset 
The OASIS RGB brain images have a dimension of (256, 256, 3) and the values of pixels were between 0 and 1 since they were normalised by a factor of 255 for efficiency - these are the inputs of the VQVAE. This data was obtained from the link: https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA. 

### Dependencies
- numpy==1.21.3
- matplotlib==3.4.3
- tensorflow==2.10.0

### Usage
The OASIS dataset images were split into train, test and validate folders, each containing subfolders of the data named as keras_png_slices_train, keras_png_slices_test and keras_png_slices_validate, respectively. In order to reuse this algorithm with different data, a similar folder structure needs to be used and the dimensions of the images may have to be modified for the convolutional neural networks to function. Training images are used for the model fitting, as with validation images - however, the latter are for reducing overfitting. Test images are used for the model predictions and image reconstructions. 

## Results
The following plots show three side-by-side comparisons of the original test OASIS brain images against the reconstructed ones, each showing their structured similarity (SSIM). 

![alt text](./images/originalvsreconstructed1.png)
![alt text](./images/originalvsreconstructed2.png)
![alt text](./images/originalvsreconstructed2.png)


## Plots of algorithm
Three metrics were used to observe the performance of testing over time. These are the overall loss, reconstructed loss, and the VQVAE loss. We see that the loss spikes intitially for the VQVAE loss (indicating overfitting) but it regularises after some epochs. Although not visable due to this spike, the recon loss behaves normally and decreases over epochs. 

![alt text](./images/testing_performance_loss.png)


## References 
https://keras.io/examples/generative/vq_vae/