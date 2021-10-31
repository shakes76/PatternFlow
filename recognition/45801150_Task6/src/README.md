# Generative model for the preprocessed OASIS brain dataset using a Vector-Quantised Variational Autoencoder

## Model architecture
The Vector-Quantised Variational Autoencoder (VQ-VAE) is a generative model proposed in 2018, that builds on the Variational
Autoencoder (VAE). Similar to a VAE, the VQ-VAE includes an encoder which takes an image and compression it down into
a latent space, and a decoder which reconstructs the latent space representation back into the original image, while
attempting to maintain as much of the original detail as possible. The difference lies in the representation of the
latent space - as opposed to a normal distribution used by a typical VAE, a Vector Quantiser is used to discretise the
latent space. This is done by creating a codebook of discrete latent vectors, with which the output of the encoder is
"snapped" to, to produce its latent representation. These vectors are snapped to the nearest discrete latent vector,
as determined by the L2 norm [1]. A diagram of this model is shown below [1]

![image](https://user-images.githubusercontent.com/55824662/139584374-4f695009-10a7-4a9e-a8b6-85d40a4e4192.png)


An autoregressive model (such as PixelCNN) can then be used to learn the prior, which can then
be used to generate high quality images [2]

## Preprocessed OASIS brain dataset
This generative model is used to create novel images of brains from the OASIS dataset. This dataset consists of 9664
training images, 544 testing images and 1120 validation images. Additionally, the images preprocessed such that they 
are all centred, and all are 256 pixels by 256 pixels [5].

## VQ-VAE results
OASIS brain images from the test set reconstructed by the VQ-VAE achieved an average structural similarity of 73%
between the test dataset and their respective reconstruction (higher than the 60% benchmark).
Visually, there is apparent blurring of the specific
details of the brains, especially around edge boundaries. Nonetheless, the overall structure of the reconstructed images
clearly resemble the original.

Below are examples of this reconstruction, with images from the test set on the left, and respective reconstructions
on the right

![image](https://user-images.githubusercontent.com/55824662/139584730-bb6a2898-5e6a-4283-abc2-f77a488180d9.png)


## OASIS brain generation results
The trained PixelCNN can be used in conjunction with the VQ-VAE to create novel images of the brain. The test script,
with the current parameters managed to generate images that looked like reasonably like brains, and were similar
to the brains provided in the OASIS dataset.

![image](https://user-images.githubusercontent.com/55824662/139584718-85259335-372a-40ce-b9ab-710368dd2439.png)


## Usage
A sample usage of this model can be demonstrated running the driver script:

```bash
$ python3 driver.py
```

This will load the OASIS brain data into memory, train the VQVAE and PixelCNN, and output
1. The average structured similarity between 10 random test images and their reconstructions will be printed to stdout
2. 10 random test images compared to their respective reconstructions will be saved to the current directory
3. 10 random generated images will be saved to the current directory.

The VQ-VAE is defined in VQ-VAE.py, along with relevant methods for trainining the model with relevant parameters. The
PixelCNN is located in 
## Dependencies
- tensorflow 2.6.0
- tensorflow-probability 0.14
- matplotlib
- Preprocessed OASIS brain dataset, found [here](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA)
## References

[1] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,” arXiv:1711.00937 [cs], May 2018, Accessed: Oct. 18, 2021. [Online]. Available: http://arxiv.org/abs/1711.00937

[2] A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves, and K. Kavukcuoglu, “Conditional Image Generation with PixelCNN Decoders,” arXiv:1606.05328 [cs], Jun. 2016, Accessed: Oct. 19, 2021. [Online]. Available: http://arxiv.org/abs/1606.05328

[3] S. Paul, "Vector-Quantized Variational Autoencoders", _keras.io_, Jul. 21, 2021. [Online]. Available: https://keras.io/examples/generative/vq_vae/. [Accessed: Oct. 18, 2021]

[4] ADMoreau, "PixelCNN", _keras.io_, May. 26, 2020. [Online]. https://keras.io/examples/generative/pixelcnn/. [Accessed: Oct. 12, 2021]

[5] D. S. Marcus, T. H. Wang, J. Parker, J. G. Csernansky, J. C. Morris, and R. L. Buckner, “Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults,” Journal of Cognitive Neuroscience, vol. 19, no. 9, pp. 1498–1507, Sep. 2007, doi: 10.1162/jocn.2007.19.9.1498.
