# Vector Quantized Variational Autoencoder (VQVAE) using OASIS dataset of brain scans
A generative model of the OASIS brain dataset using VQVAE to reconstruct a "reasonably clear images" with Structured Similarity (SSIM) of over 0.6. Examples of reconstructed images and a graph of loss rates are provided. 

## Algorithm 
### VQVAE model
The VQVAE is a variational autoencoder that operates on a discrete latent space as opposed to a continious latent space in the regular VAEs. The advantage of this is simplifying the optimization. The VQVAE does this by using the idea of a discrete "codebook" that takes the continuous embeddings (sampled from normal distribution) and encoded outputs and finds their discrete difference, which is subsequently put through the decoder and trained for image reconstruction. The encoder is a convolutional network using downsampling, whereas the decoder uses upsampling in its convolutional network. 


## Implementation 
### Dataset 
The OASIS brain images have a dimension of (256, 256, 3). 



## Results




## Plots of algorithm



## References 


