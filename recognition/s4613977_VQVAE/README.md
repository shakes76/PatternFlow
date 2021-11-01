# VQVAE on the OASIS dataset

We're here trying to train a generative VQVAE to generate images from the OASIS dataset with an SSIM of at least 0.6.

The VQVAE does the encoding and the quantising, but also the decoding. A PixelCNN is used alongside it to perform prior training for random image generation (with a quantised input such as the VQVAE would produce)

## I. Data

The datasets I used were those in the link on blackboard. I intentionally discarded the segmentation datasets. The data is loaded thanks the load_data() function which also normalises the images so we don't have to

## II. Models implemented

Most of the code was adapted from the corresponding Keras tutorial (https://keras.io/examples/generative/vq_vae/).

### a) VQ-VAE

#### The model

Class VQ : the vector quantiser class, which, as its name may suggest, implements the codebook. Used 512 codebook vectors here.

Function encoder : implements the encoder. Opted for 4 convolutional layers with, in that order, 256, 128, 64 and 128 filters (the last number corresponds to the dimension of the codebook vectors)

Function decoder : implements the decoder, very much just the symmetric to the encoder

Function vqvae : builds the VQVAE with calls to VQ, encoder, and decoder. The end product had over 955,000 parameters:

![Capture](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture.PNG)

#### Training

Trained for 20 epochs with batch size of 32. The weights were saved under the name "VQVAE".

#### Results

If we take a look a a few encoded and decoded test images, that's what we get for 2 of them :

![Capture_1](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_1.PNG)

Not too bad, but still sort of blurry, the details are lacking. As for the corresponding codes :

![Capture_2](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_2.PNG)

Obtained a mean SSIM of 0.8682193400121998 on the test set for the, as shown on the picture below:

![ssim](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\ssim.PNG)

### b) PixelCNN

#### The model

This one was far deeper than the VQ-VAE. It would take too long to sum up its characteristics, but I built it with 16 residual blocks and 16 convolutional layers. Here's the bottom of its summary:

![Capture_3](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_3.PNG)

#### Training

I trained it on 100 epochs, with batch size 32 and validation split 0.1 :

![Capture_4](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_4.PNG)

#### Results

#### Random image generation with priors



![Capture_5](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_5.PNG)

As we can see, a few generated samples actually look like brains (I wish I found a way to show more on here but hopefully you can find everything on the notebook).

Generating test images with the test set encodings yielded the following (I randomly printed out 10 and below are 4 of those 10):

![Capture_6](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_6.PNG)

As for the SSIM, it is as follows :

![Capture_7](C:\Users\Sofia\OneDrive\Desktop\UQ\MSc\S2\COMP3710\Report\PatternFlow\recognition\s4613977_VQVAE\README.assets\Capture_7.PNG)

Not how it looks extremely similar to the VQ-VAE decoded test image. The two are actually different, but the difference only starts at the 11th decimal place.

## III. Dependencies

tensorflow 2.4

tensorflow-probability 0.12.1

numpy (any version that's compatible with tensorflow 2.4)

imageio (data loading)

skimage.metrics (for ssim)

os (used the listdir function to load the images)

## IV. Additional Notes

I imported both the model weights and pushed them on the repository (they're named "VQVAE" and "PixelCNN"). But you're more than welcome to rerun the training if you wish to.

I also included the notebook. The raw code in the Code.py file, and the notebook in the Code.ipynb file.
