# Vector-Quantised Variational AutoEncoder (VQ-VAEs)
The Vector-Quantised Vriational AutoEncoder was proposed by Oord et al, in the paper titled [Neural Discrete Representational Learning](https://arxiv.org/pdf/1711.00937v2.pdf). It describes combining the Variational Auto-Encoder with Vector Quantisation to learn a discete latent representation of data (rather than hard-coding it).

**Latent Representation**: A data's latent space is described as an underlying representation of how features within data are distributed - this is akin to the components derived from PCA analysis. [[1]](#References#1)

## VQ-VAE Component
In this implementation, the VQ-VAE model is comprised of two key sub-models - the [Encoder](#encoder) and [Decoder](#decoder), as well as a [Vector Quantisation](#vector-quantisation) component. In short, the encoder is responsible for transforming the inputs (in this case, X-Ray images of brains) into the latent space. Likewise, the decoder is responsible for transforming latent space samples (that is, the output of the encoder), and reconstructing images from it.

For this explanation, I will use the following variables:
$\begin{aligned}b:\ &\text{Batch Size}\\h:\ &\text{Image Height}\\w:\ &\text{Image Width}\\c:\ &\text{Number of Channels in Input Image}\\d:\ &\text{Latent Size}\end{aligned}$
### Encoder
The encoder is mathematically represented by the equation $z=f(x)$ and takes in an input image (or batch of images) of shape $(b, h, w, c)$. Through a series of convolutional layers, it transforms the input image into the latent space.

### Decoder
The decoder is mathematically represented by the equation $\hat{x}=g(f(x))$ which takes the latent embedding $f(x)$ of a particular image $x$ and attempts to reconstruct it, producing the image $\hat{x}$.

### Vector Quantisation
When using traditional Variational AutoEncoders (VAEs), the latent encodings learned don't necessarily group data with similar features together, and the bounds of the latent space (that is, the minima and maxima of each value) are far away from the mean. In addition to this, the latent space is described by continuous variables, which further increases the complexity of the problem.

When using VQ-VAEs, we introduce a discrete codebook which allows us to discretise the continuous embedding. The codebook is a list of vectors associated with a corresponding latent space axis. By comparing the continuous latent encoding with all vectors in the codebook, we can select the vector that is closest in euclidean distance to the continuous latent encoding produced by the encoder. This is mathematically represented as:
$$ z_q(x)= \argmin_{i} |z_e(x)-e_i|$$
Where
$\begin{aligned} x:\ &\text{Input Image to Encoder}\\e_i:\ &\text{ith codebook vector}\\ z_q(x):\ & \text{Resulting quantised vector, input to decoder}\end{aligned}$

If the discretised latent encoding is a good representation of the data, then passing it (the discretised encoding) through the decoder should yield a reconstructed image $\hat{x}$ that is similar to the original image passed to the encoder $x$.

### VAE - Summary
If the decoder is able to accurately reconstruct the image in structure and shape then this implies that the latent space is a good representation of the data. To improve this, we use a **reconstruction loss** which is the difference between the original and reconstructed image as the loss function in our training.
$$ L(x, \hat{x}) = |x-\hat{x}|$$


## Generative Component








# References
[1] [Understanding VQ-VAE (DALL-E Explained Pt. 1) by Snell, Charlie](https://libraryguides.vu.edu.au/ieeereferencing/gettingstarted)
