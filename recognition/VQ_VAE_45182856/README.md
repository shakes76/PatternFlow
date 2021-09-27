# VQ-VAE for OASIS dataset

## Overview

This project uses Vector-Quantized Variational Autoencoder model ([van den Oord et al,. 2017](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)) to learn the manifolds of MRI brain images in OASIS dataset as a means to understand the pattern of the data. Specifically, once the manifolds are learnt, one can visualize these manifolds and make use of the generative ability as well as the decoder of the model to visualize unseen images, which is useful for both model interpretation and data understanding. Regarding the VQ-VAE architecture (Fig. 1), it is comprised of an encoder (i.e., this part encodes the original data to a more compact representation), codebook (i.e., list of latent vectors to describe the manifolds of the original data), and a decoder (i.e., this part uses the manifolds to reconstruct images). About the dataset used in this project, Fig. 2 depicts a small subset of images in this dataset.

<p align='center'>
    <img src='images/VQ-VAE.png'
</p>

 <p align='center'>
     Fig. 1. The architecture of VQ-VAE (Image source: <a href="http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf">van den Oord et al,. 2017</a>
 </p>

<p align='center'>
    <img src='images/sample_images.png'
</p>

 <p align='center'>
     Fig. 2. Sample images extracted from the OASIS dataset
 </p>







