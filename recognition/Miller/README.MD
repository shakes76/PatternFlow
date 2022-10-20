Vector Quantized Variational Auto-encoder(VQ VAE Model)
Oasis Brain Data Set
This program creates a create a generative model of the OASIS brain data set that has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6.

Description of VQ VAE Algorithm:
A standard VAE (encoder->decoder) uses a continous latent space that was sampled using gaussain distribution; this makes it hard to leatn continuous distribution with a gradient descent. In comparison, VQ VAE (encoder-> VQ layer-> decoder) ues discrete latent space; and consists of three parts:

1. Encoder -> Convolutional network to downsample the features of an image

2. Latent space -> Discrete "codebook" that describes the latent space. 
    codebook consists of n latent embedding vectors of dimension D each
    each code represents the distance between each embedding and encoded output+

VQ VAES:
    disccrete latent space
    optimised by using discrete "codbook" -> made by discreting dist between continuous embedding and encoded outputs, the ndiscrete codewords sent to decoder which trained to generate reconstructered samples.

    Generative model based on VAE. 
    Aims to make latent space discrete using VQ techniques

    cons: loses the "easy latent sampling" propery of VAES. 2 stage training required to learn fitting categorical prior
    + training objective not correspond to bound on log-likelihood amnymore


