# Vector Quantized Variational Autoencoder (VQVAE) using OASIS dataset of brain scans
A generative model of the OASIS brain dataset using VQVAE to reconstruct a "reasonably clear images" with Structured Similarity (SSIM) of over 0.6. Examples of reconstructed images and a graph of loss rates are provided.

## Algorithm 
### VQVAE model
The VQVAE is a variational autoencoder that operates on a discrete latent space as opposed to a continious latent space in the regular VAEs. The advantage of this is simplifying the optimization. The VQVAE does this by using the idea of a discrete "codebook" that takes the prior continuous embeddings (sampled from uniform distribution) and encoded outputs and finds their discrete difference, which is subsequently put through the decoder and trained for image reconstruction. The encoder is a convolutional network using downsampling, whereas the decoder uses upsampling in its convolutional network. In this algorithm, the encoder and decoder both have three convolutional layers, whereby the decoder reverses the encoders layers. 

## Implementation 
### Dataset 
The OASIS RGB brain images have a dimension of (256, 256, 3) and the values of pixels were between 0 and 1 since they were normalised by a factor of 255 for efficiency - these are the inputs of the VQVAE. This data was obtained from the link: https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA. A total of 11334 images are used with 9664 (85%) belonging to the train set, 1120 (10%) to the validation set and 544 (5%) to the test set.

### Dependencies
These modules are required to run this algorithm:
- python==3.7.9
- numpy==1.21.3
- matplotlib==3.4.3
- tensorflow==2.10.0

### Usage
First extract the OASIS data from the link above and chose a location for the data. The OASIS dataset images were split into train, test and validate folders, each containing subfolders of the data named as train/keras_png_slices_train, test/keras_png_slices_test and validate/keras_png_slices_validate, respectively - and the segmentation images were discarded. In order to reuse this algorithm with different data, a similar folder structure needs to be used and the dimensions of the images may have to be modified for the convolutional neural networks to function. Alter the "path' variable in dataset.py depending on your file location. Nevertheless, training images are used for the model fitting, as with validation images - however, the latter are for reducing overfitting. Test images are used for the model predictions and image reconstructions. Run train.py to train model and test.model for predictions of reconstructed images. 

## Results. 
The following plots show three side-by-side comparisons of the original test OASIS brain images against the reconstructed ones. The first reconstruction has a SSIM of 0.84565616, second reconstruction with SSIM of 0.83053225, and 0.85235796 for the third reconstruction. Hence, observe that the average SSIM is greater than 0.6. Each image reconstruction has an SSIM to its original within a range of 0.8 to 0.9. 

![alt text](./images/originalvsreconstructed1.png)
![alt text](./images/originalvsreconstructed2.png)
![alt text](./images/originalvsreconstructed2.png)


## Plots of algorithm
Three metrics were used to observe the performance of testing over time. These are the overall loss, reconstructed loss, and the VQVAE loss. We see that the loss spikes intitially for the VQVAE loss (indicating overfitting) but it regularises after some epochs. Although not visable due to this spike, the recon loss behaves normally and decreases over epochs. 

![alt text](./images/testing_performance_loss.png)


## References 
https://keras.io/examples/generative/vq_vae/