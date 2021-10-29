# StyleGAN Implementation

## Preview
This project is an implementtion on StyleGAN networks based on
the paper ...

![img.png](img.png)

"We first map the input to an intermediate latent space W,
which then controls the generator through adaptive
instance normalization (AdaIN) at each convolution layer.
Gaussian noise is added after each convolution, before
evaluating the nonlinearity. Here “A” stands for a learned
affine transform, and “B” applies learned per-channel scaling fac-
tors to the noise input. The mapping network f consists of 8 lay-
ers and the synthesis network g consists of 18 layers—two for 22
each resolution (4 − 1024). The output of the last layer is
converted to RGB using a separate 1 × 1 convolution, similar to
Karras et al. [30]. Our generator has a total of 26.2M trainable
parameters, compared to 23.1M in the traditional generator."
