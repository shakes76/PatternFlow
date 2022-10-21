# Vector Quantized Variational Auto-encoder(VQ VAE Model)

In this report, a generative model of the Vector Quantized Variational AutoEncoder (VQ VAE) was used to generate reconstructed images of the OASIS brain data set that are "reasonably clear" and have a Structured Similarity (SSIM) of over 0.6. The VQ VAE was adapted using tensorflow keras.

#### Description of VQ VAE Algorithm
![](https://miro.medium.com/max/1400/1*yRdNe3xi4f3KV6ULW7yArA.png)
>Figure 1: Graphical representation of a VQ-VAE network.

A standard VAE (encoder->decoder) uses a continous latent space that is sampled using gaussain distribution; this makes it hard to learn a continuous distribution with a gradient descent. In comparison, VQ VAE uses a discrete latent space; and consists of three parts as seen above:

1. Encoder:
    * Convolutional network to downsample the features of an image
2. Latent Space:
    * Codebook consists of n latent embedding vectors of dimension D each
    * Each code represents the distance between each embedding and encoded output (euclidean distance) ->outputs embeded vector
    * feed closest encoder output to codebook as input to decoder
3. Decoder:
    * Convolutional network to upsample and gnerate reconstructed samples.

#### ==============Oasis Brain Data Set==============
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRl7czOsj3uzWRQ6NT2ofed7QBsKiqrUq6Bsw&usqp=CAU)
>Figure 2: Comparison of an image stored in the train vs test data sets

The Oasis MRI Dataset cobtains 9664 training images, 544 test images and 1120 validation images.  An example of train and test data is shown above. The images are preloaded into a file location and from there extracted into processing for use.

##### Data Pre-Processing

Before the data was used, it was normalised through residual extration and rescaling. This makes it easier to compare the distributions with different means and scales to maintain the shape of the distribution. 

## ==============Training==============

The three data groups - train, test, and validate are split 0.85/0.1/0.05. The training set contains the most images so the model has enough information to learn from to produce accurate reconstructions later. The test set is used to validate these reconstructions. The validation set is not required, as the model is judged by the quality of the reconstructons on the test set. The model is trained with 5 epochs on a batch size of 128.

![](/recognition/s4581053%20VQVAE%20OASIS/Images/VQVAE%20LOSS%20GRAPH.png)

>Figure 3: VQVAE Reconstruction loss over 5 epochs

## ==============Results==============

Below are 5 randomly chosen reconstructed images and its comparison to a test image

NOTE: The below images and Avergae SSIM rating was originally run on 15 epochs. NOT 5.

![](/recognition/s4581053%20VQVAE%20OASIS/Images/Reconstructed%20Image%201.png)

![](/recognition/s4581053%20VQVAE%20OASIS/Images/Reconstructed%20Image%202.png)

![](/recognition/s4581053%20VQVAE%20OASIS/Images/Reconstructed%20Image%203.png)

![](/recognition/s4581053%20VQVAE%20OASIS/Images/Reconstructed%20Image%204.png)

![](/recognition/s4581053%20VQVAE%20OASIS/Images/Reconstructed%20Image%205.png)

>Figure 4: 5 randomly chosen reconstructed images and their ssim value compared to it original image

The reconstructed images achieved a mean Structured Similarity of 73.4

## Dependencies
* Python 3.7
* TensorFlow 2.6.0
* Numpy 1.19.5
* matplotlib 3.2.2
* Pillow 7.1.2
* os
* Pre-processed OASIS MRI dataset (accessible at https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ/download).

## References
[1] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, 2018. Neural Discrete Representation Learning. [Online]. Available at: https://arxiv.org/pdf/1711.00937.pdf.

[2] Paul, S., 2021. Keras documentation: Vector-Quantized Variational Autoencoders. [online] Keras.io. Available at: https://keras.io/examples/generative/vq_vae/.

[3] https://github.com/shakes76/PatternFlow/tree/master/recognition/MySolution

[4] https://keras.io/examples/generative/pixelcnn/
