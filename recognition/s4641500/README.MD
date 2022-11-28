# OASIS Vector-Quantized Variational Autoencoder
This is an algorithm for reconstructing OASIS images utilising a VQ-VAE. The benefit of using a VQ-VAE over a standard variational autoencoder is the reduction in time taken to train and evaluate the model. Using this VQ-VAE it is possible to encode and generate new 2d slices of human brains. This is especially meaningful for detecting physical anomalies. Because the VQVAE is trained to learn only the most important features of brains, a reconstruction of a diseased or damaged brain image would differ greatly from the input (ie, look more "normal"). This becomes very important for the quick identifiication and classification of physical issues within the human brain and swift diagnoses of concerns.

![](vae.png) <br />
(Image from [lucas-bechberger.de](https://lucas-bechberger.de/2018/12/07/what-is-a-%CE%B2-variational-autoencoder/))

Vector-Quantized Variational Autoencoders work by using vector quantization to obtain a discrete latent representation. They stand out from standard VAEs, functionally, through incorporation of vector-quantization and how the encoder outputs discrete rather than continuous codes and priors are learned rather than constant. Using the VQ method allows the model to circumvent issues of posterior collapse - where the latents are ignored when they are paired with a powerful autoregressive decoder - typically observed in the VAE framework. Using such representations with a prior, the model can generate high fidelity output.

## Dependencies
This code was written and tested using tensorflow version 2.10.0 and tensorflow-probability version 0.18.0, numpy 1.23.3, Pillow 9.2.0 for images, matplotlib 3.6.1 for plots and keras version 2.10.0 for models.

The program was written within a conda environment with cudatoolkit version 11.2.2 and cudnn version 8.1.0.77 installed, and on a Windows system.

## Results
![](example-reconstruction.png)

As we can see, the priors constructed from an input after training a model can be used to recreate what the VQ-VAE "thinks a brain should look like" based on the priors code. This impression of what a brain looks like can be trained from 2d image slices of healthy or unhealthy brains, allowing reconstruction in such a way that an unhealthy brain can be reconstructed to look healthy or a healthy brain to look unhealthy - and this can be used to assist detection of potential problems in medical circumstances.