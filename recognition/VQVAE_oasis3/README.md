## VQVAE2 Implementation for OASIS Dataset

#### VQVAE2 Algorithm
This project sought to implement a VQVAE-based image generator for the OASIS3 dataset. To achieve this, the algorithm used by Razavi et al in *'Generating Diverse High-Fidelity Images with VQ-VAE-2'* was adapted into a Tensorflow Keras implementation that worked on the monochrome image data. The final form of this algorithm as implemented here is shown below.

![vqvae_diagram](vqvae_diagram.png)

The filter values given in the figure above resulted in a Structured Similarity (SSIM) score of 0.867. This was achieved by training the algorithm for 400 epochs, and the step-wise results of this training are shown in this graph.

**INSERT GRAPH HERE**

