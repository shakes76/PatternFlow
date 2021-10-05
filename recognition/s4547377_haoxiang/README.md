<h1 style="color:Tomato;">Implement the style based GAN model with pytorch.</h1>

<h1>This repository contains the pytorch implementation of the Style-Based Generator Architecture for Generative Adversarial Networks.</h1>

Abstract: We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.

If you know nothing about the GAN model, please have a look at the file demo2Part3.ipynb, which contains the normal GAN model implementation with the pytorch.

<h1 style="color:Tomato;">Resources:</h1>

The origin image sets can be downloaded via : https://www.oasis-brains.org/

<h1>System requirements:</h1>
Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.

64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.

Pytorch 1.9.0 or newer with GPU support.

Pytorch build channel py3.8_cuda11.1_cudnn8_0 or newer with GPU support.

One or more high-end NVIDIA GPUs with at least 11GB of DRAM. We recommend NVIDIA DGX-1 with 8 Tesla V100 GPUs.

NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.


<h1 style="color:Tomato;">Explanation:</h1>

The style based GAN model is an extension to the GAN architecture that proposes large changes to the generator model, including the use of a mapping network to map points in latent space to an intermediate latent space, the use of the intermediate latent space to control style at each point in the generator model, and the introduction to noise as a source of variation at each point in the generator model.
