“README.MD” to sufficiently document your project (see Section 6).



Create a generative model of one of the OASIS brain, ADNI brain or the OAI AKOA knee data set (see
Appendix for links) using a VQVAE [11] or VQVAE2 [12] that has a “reasonably clear image” and a Structured Similarity (SSIM) of over 0.6. [Hard Difficulty]

Brief Description:
Vector Quantized Variational Autoencoder (VQ-VAE)
standard VAEs:
    latent space is continous + sampled using gaussian distribution ------> Brush up on latent space and gaussian distribution
    with gradient descent, becomes harder to learn continous distribution --------> What is gradient descent
VQ VAES:
    disccrete latent space
    optimised by using discrete "codbook" -> made by discreting dist between continuous embedding and encoded outputs, the ndiscrete codewords sent to decoder which trained to generate reconstructered samples.

    Generative model based on VAE. 
    Aims to make latent space discrete using VQ techniques

    cons: loses the "easy latent sampling" propery of VAES. 2 stage training required to learn fitting categorical prior
    + training objective not correspond to bound on log-likelihood amnymore