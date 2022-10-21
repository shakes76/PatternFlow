# COMP3710 Open Source Pattern Recognition Project
## Christian Binggeli, 45856682, GNUNotUsername

## Background
The vector-quantised variational autoencoder (VQ-VAE) is a variation of the variational autoencoder (VAE); which itself is a variation of the autoencoder (AE). The typical autoencoder consists of two halves; the encoder and the decoder. The encoder makes use of convolutional layers to reduce an image (high dimensional input) into a latent vector (low dimensional), part of a latent space. The decoder then makes use of convolutional transpose layers to reconstruct a latent vector into an image relatively similar to the original.
![autoencoder](https://user-images.githubusercontent.com/99316773/197157234-fcee6029-d887-41ab-abbd-b2e4755e4fa2.png)
Reference [2]

A VAE differs from a regular AE by allowing for the insertion of new latent latent vectors following a normal distribution into the latent space. The decoder can then use these new latent vectors to generate new images. In essence, this allows for the creation of images which never existed before from random noise. If the latent space were two dimensional, images generated from select 2D vector would be viewable in a 2D manifold. An example of this (from my own COMP3710 Demo 2) is shown below:
![manifolds](https://user-images.githubusercontent.com/99316773/197157327-c9f4c2d0-fdf0-49bc-b605-c826fbfcf458.png)
A VQ-VAE differs again from this by using a vector quantisation layer to force the VQ-VAE to learn discrete latent vectors instead of continuous latent vectors. The set of all discrete vectors learned by the VQ-VAE is known as the vector codebook, or set of codebook vectors.

A pixel-convolutional neural network (PCNN) can be used to generate new discrete latent vectors. The PCNN trains alongside the VQVAE's encoder to learn to generate encodings, then uses a separate image set to generate encodings not before seen by the VQVAE. These encodings are then sent to the VQVAE's decoder and are used to generate new images.

The VQVAE and PCNN architecture were inspired by reference [1] and research by Oord, Kalchbrenner et.al [4].

## Datasets
The OASIS brains dataset was used in this project. Each image was scaled by a factor of 255 and offset by -0.5. This was done in order to normalise the pixel values to within the range of [-0.5, 0.5]. Each image also was reduced in size from 256x256 pixels to only 80x80 pixels. The GPUs within the machines used for this project could not handle full sized images and so these had to be scaled down.

## Training Process
The VQVAE/PCNN model was trained on a machine equipped with a NVidia 1660Ti Mobile GPU and an Intel Core i7-9750H CPU running Arch Linux, kernel version 6.0.2. Both the VQVAE and the PCNN were trained for 100 epochs. With this hardware and an apropriate cuDNN installation, epoch of the VQVAE took seven seconds and each epoch of the PCNN took three seconds. This resulted in an overall training time of 16 minutes 40 seconds.

The OASIS dataset originally contains 9,664 training images, 544 testing images and 1,120 validation images. Instead of using the existing validation set, the training set was
partitioned into training and testing splits. This was done at a ratio of 0.9:0.1. The batch size used was 128.

Graphs of the VQVAE's loss, the PCNN's loss and the PCNN's accuracy across the epochs are shown below:

![vqvae loss](https://user-images.githubusercontent.com/99316773/197157602-bd40e336-9e66-483b-8af4-74d038bb67d5.png)
![pcnn loss](https://user-images.githubusercontent.com/99316773/197157638-af0d3d0f-64af-4df8-b0e2-357696e402ac.png)
![pcnn acc](https://user-images.githubusercontent.com/99316773/197157661-6622977b-ab9e-4a3f-8b1a-eedde5ee4f7d.png)

# Results
Firstly, the VQVAE was trained to encode and decode brain images by itself. It was decided that a structured similarity (SSIM) of at least 0.6 is desired for the VQVAE. The results of the encoding and reconstruction of five test images are shown below. As visible, this achieved an average SSIM of 0.84 rounded to two decimal places. The reconstructions are reasonably clear and recognisable to their respective original images.
![encodedecode](https://user-images.githubusercontent.com/99316773/197157805-15ad152d-9a6d-463c-a4bb-41d6474b5fa1.png)
The PCNN was then trained to create discrete latent vectors from input images. Some discrete latent encodings are visible below. These two are also recognisable to their respective original images, except vastly more pixellated.
![good pcnn](https://user-images.githubusercontent.com/99316773/197158013-dbc1cd5a-9a66-4d38-a6c7-5d3f629c25be.png)
The discrete latent vectors generated from the PCNN were then fed through the VQVAE decoder to produce new brain images. Results of this are visible below. These are less clear than images of the original OASIS set, but are still quite clear.
![GOOD CODES](https://user-images.githubusercontent.com/99316773/197158072-be7ddfb0-ce73-433c-b4f4-6518a7a63e12.png)

# Dependencies
* python 3.10.8
* keras 2.9.0
* matplotlib 3.5.3
* numpy 1.23.2
* Pillow 9.2.0
* tensorflow 2.9.1
* tensorflow-probability 0.17.0
* glob
* Oasis Dataset [4]

Based from $pip list

# References
*[1] https://keras.io/examples/generative/vq_vae/
*[2] https://miro.medium.com/max/720/1*qFzKC1GqOR17XaiQBex83w.png
*[3] https://arxiv.org/pdf/1606.05328.pdf
*[4] https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA
