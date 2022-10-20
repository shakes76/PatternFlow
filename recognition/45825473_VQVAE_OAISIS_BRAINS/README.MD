# Implementation of generative VQVAE for preprocessed OASIS

## VQVAE & its model architecture

The model architecture for the VQVAE is based on the concept of an autoencoder that maintains a discrete latent space rather than continuous. Thus the VQVAE contains an encoder model that generally performs convolutions on the given input, a latent space where was inspired by the keras tutorial (CITATION HERE). but ultimately adjusted and then hypertuned to suit the OASIS data set. This involved making the following adjustments:

- Doubling the number of filters in the encoder layer and decoder layer of the VQVAE.
- Increasing the kernal size of the PixelCNN to 9.
- Increasing the number of residual blocks and convolution pixel layers from 2 each to 6 each.
- Training the VQVAE for 75 epochs
- Training the PixelCNN for 250 epochs

The overall structure of the VQVAE can be visualised below

![image](https://raw.githubusercontent.com/Adam99115/PatternFlow/topic-recognition/recognition/45825473_VQVAE_OAISIS_BRAINS/images/vqvae-model-diagram.png)

## Preprocessed OASIS brain dataset

The OASIS data used in this problem to train the generative VQVAE was downloaded . IT consists of two different twos - a sliced images set divided into validation, training and test partitions. For this problem I will be us ing, the training partition conists of 9664 train images, 544 test images and 1120 validation images.

Although initial attempts to train the model using the original images (256x256 pixels), it was found that this was consuming too much memory and hence the images for each partition were further processed by downscaling the size of each item into 128x128 images. This not only sped up training times per epoch but also didn't over allocated memory on the machine I was training the VQVAE with.

## Results

Overall, the model produced reasonable results with both reconstructed and generated images produced by the decoder presenting brain-like images from both decoding the test test but also generating sample images from the pixel cnn and decoding these.

### Brain reconstruction

![image](https://raw.githubusercontent.com/Adam99115/PatternFlow/topic-recognition/recognition/45825473_VQVAE_OAISIS_BRAINS/images/ReconstructedBrains.PNG)

### Loss & SSIM

![image]

### Brain Generations

![image](https://raw.githubusercontent.com/Adam99115/PatternFlow/topic-recognition/recognition/45825473_VQVAE_OAISIS_BRAINS/images/GeneratedBrains.PNG)

## Dependencies

    The list of dependencies required for this implementation are as follows:

    - python 3
    - tensorflow 2.6.0
    - tensorflow-probability 0.14.0
    - numpy 1.21.3
    - matplotlib 3.4.3

## Usage

!ENSURE YOU HAVE THE OASIS slices folder located in the same folder as 45825473_VQVAE_OASIS_BRAINS

To train the model based on the VQVAE model architexture in

```bash
$ python3 train.py
```

If you would like to (Please ensure the folder containing your VQVAE model and it is labelled "VQVAE_Model")

```bash
$ python3 predict.py [-m <PathToPreBuiltVQVAEModel>]
```

## References

[1] S. Paul, "Vector-Quantized Variational Autoencoders", _keras.io_, Jul. 21, 2021. [Online]. Available: https://keras.io/examples/generative/vq_vae/ [Accessed: Oct. 16, 2022]

[2] ADMoreau, "PixelCNN", _keras.io_, May 26, 2020. [Online]. Available: https://keras.io/examples/generative/pixelcnn/ [Accessed: Oct. 16, 2022]

[3] Sieun Park "An overview on VQ-VAE: Learning Discrete Representation Space" Apr. 4, 2021. Available: https://miro.medium.com/max/720/1*miNfFc9qT5PrS7ectJa_kw.png [Accessed: Oct. 17, 2022]
