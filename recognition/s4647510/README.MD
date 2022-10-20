# VQVAE using OASIS brain dataset

This project is a generative model of the OASIS brain dataset using a Vector-Quantized Variational Autoencoder (VQVAE) and a PixelCNN for image generation. 

## Algorithm Description

### VQVAE

The VQVAE is a modified version of a standard Variational Autoencoder where it operates on a discrete latent space rather than a continuous distribution. It is able to generate high quality novel brain slice images based off the OASIS dataset.

Firstly, the VQVAE involves an encoder convolutional network that creates a downsampled code for each image. The VQVAE also involves a convolutional network decoder, which transforms a code back into an image.
The latent space in this autoencoder is a trainable codebook which is quantised rather than being a continuous normal distribution.

![VQVAE Diagram](https://user-images.githubusercontent.com/41940464/196821144-fa41993c-5bfb-48e0-8837-b74925c2fde0.png)

### PixelCNN

Once the VQVAE has been trained, the PixelCNN can use the VQVAE to generate images one pixel at a time using trained priors, with each next pixel value being determined by the currently generated pixels.
The PixelCNN uses two types masked convolutional layers to mask unpredicted pixels.

![Pixel CNN Mask](https://user-images.githubusercontent.com/41940464/196822113-d99c0f97-d382-40b7-9243-7118a122e5ef.png)

Mask type A is used on the first convolutional layer and only allows previously generated pixels to be seen.
Mask type B is used on all subsequent layers and allows previously generated pixels and the currently generated pixel to be seen.

The PixelCNN is trained to learn a probability distribution. Once trained, codes can be decoded by the VQVAE to generate images.

| ![PixelCNN Diagram](https://user-images.githubusercontent.com/41940464/196823793-81c7e380-6e96-4770-aab4-209b2a4700f2.png) | 
|:--:| 
| https://towardsdatascience.com/autoregressive-models-pixelcnn-e30734ede0c1 |

## Dependencies

tensorflow >= 2.10
numpy >= 1.23.4
matplotlib >= 3.6.1
PIL >= 9.2

## Training

### Dataset Preprocessing

The OASIS brain dataset was used with 1154 training images and 544 testing images. 20% of the training images were used for validation of the PixelCNN.

The images were 256x256 grayscale images with 256 pixel value depth which was normalised between 0 and 1 in preprocessing. Images were converted to numpy arrays for processing.

![case_011_slice_12 nii](https://user-images.githubusercontent.com/41940464/196826679-22bf8a94-a935-47f6-96dd-28548558c103.png) ![case_014_slice_21 nii](https://user-images.githubusercontent.com/41940464/196826695-981a8afb-213a-4735-94c0-e523108ec018.png) ![case_019_slice_23 nii](https://user-images.githubusercontent.com/41940464/196826698-3a5e1b31-943e-418c-819b-2fa8ac67ed83.png)

### VQVAE Training

The VQVAE was trained with the training images for 300 epochs and ended with a reconstruction SSIM of approximately 0.92.
It was trained using an Adam optimiser and a batch size of 8.

![training_plot](https://user-images.githubusercontent.com/41940464/196826876-3c6785a8-8b7d-43b7-8d71-f55a5438a835.png)

An example encoding and reconstruction is shown below:

![codebook_9](https://user-images.githubusercontent.com/41940464/196827599-b57436b2-d712-4ff8-adaf-6fe81a29945f.png)
![reconstruction_9](https://user-images.githubusercontent.com/41940464/196827313-7fa4f106-a178-4580-9403-260c2ba9133f.png)

### PixelCNN Training

The PixelCNN was trained and validated with an 80:20 split in the training images for 10,000 epochs, with the learning shown below.
It was trained using an Adam optimiser, a learning rate of 0.0003, a batch size of 32, and a sparse categorical crossentropy loss function.

![cnn_accuracy_plot](https://user-images.githubusercontent.com/41940464/196828147-357f9a81-cd0e-4ad9-b51c-18578745beb7.png)
![cnn_loss_plot](https://user-images.githubusercontent.com/41940464/196828151-58d0c943-4a9c-4271-8c8e-2e6a09516be1.png)

While the validation loss increases and the validation accuracy stagnated after 100 epochs, the quality of the generated images increased over all epochs.
A comparison of an image generated at 100 epochs and 10,000 epochs is shown below:

![decoded_1](https://user-images.githubusercontent.com/41940464/196830791-172061d4-fa23-420c-bdb4-955bf83f262c.png)
![decoded_0](https://user-images.githubusercontent.com/41940464/196831266-f5bef2c2-4dcf-4211-bec7-2a0dbf44dcc4.png)

### Results

Many images were generated with an SSIM score > 0.6, with some shown below:

![decoded_8](https://user-images.githubusercontent.com/41940464/196828872-0ad87d5f-92e7-4866-aedd-34b26be900e1.png)
![decoded_1](https://user-images.githubusercontent.com/41940464/196828873-153c71dd-c6c4-439b-8406-85658d4587e9.png)
![decoded_3](https://user-images.githubusercontent.com/41940464/196828875-10a160e6-6e18-4510-a726-7a31b20b1828.png)

## References
1. A.v.d. Oord, O. Vinyals, and K. Kavukcuoglu, “Neural Discrete Representation Learning,”
arXiv:1711.00937 [cs], May 2018, arXiv: 1711.00937. [Online]. Available: http://arxiv.org/abs/1711.00937
2. P. Esser, R. Rombach, B. Ommer, "Taming Transformers for High-Resolution Image Synthesis",
arXiv:2012.09841 [cs.CV], Dec 2020, arXiv: 2012.09841. [Online]. Available: https://arxiv.org/abs/2012.09841
3. https://keras.io/examples/generative/vq_vae/
4. https://keras.io/examples/generative/pixelcnn/
