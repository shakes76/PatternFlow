# VQVAE on the OASIS dataset

## I. Data

The datasets I used were those in the link on blackboard. I intentionally discarded the segmentation datasets. The data is loaded thanks the load_data() function which also normalises the images so we don't have to

## II. Models implemented

Most of the code was adapted from the corresponding Keras tutorial (https://keras.io/examples/generative/vq_vae/).

### a) VQ-VAE

#### The model

Class VQ : the vector quantiser class, which, as its name may suggest, implements the codebook. Used 512 codebook vectors here.

Function encoder : implements the encoder. Opted for 4 convolutional layers with, in that order, 256, 128, 64 and 128 filters (the last number corresponds to the dimension of the codebook vectors)

Function decoder : implements the decoder, very much just the symmetric to the encoder

Function vqvae : builds the VQVAE with calls to VQ, encoder, and decoder. Assembles them together to get the model. Below is a summary of the end product.

![image-20211101004202872](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101004202872.png)

#### Training

Trained for 20 epochs with batch size of 32. The weights were saved under the name "VQVAE".

#### Results

If we take a look a a few encoded and decoded test images, that's what we get for 3 of them :

![image-20211101004816260](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101004816260.png)

Not too bad, but still sort of blurry, the details are lacking. As for the corresponding codes :

![image-20211101004910123](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101004910123.png)

Obtained a mean SSIM of 0.8682193400121998 on the test set for the, as shown on the picture below:

![image-20211101030823178](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101030823178.png)

### b) PixelCNN

#### The model

This one was far deeper than the VQ-VAE. It would take too long to sum up its characteristics, but I built it with 16 residual blocks and 16 convolutional layers. Here's the bottom of its summary:

![image-20211101034250284](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101034250284.png)

#### Training

I trained it on 100 epochs, with batch size 32 and validation split 0.1 :

![image-20211101034407798](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101034407798.png)

#### Results

#### Random image generation with priors

![image-20211101034501097](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101034501097.png)

![image-20211101034519840](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101034519840.png)

As we can see, a few generated samples actually look like brains, and few others, not so much... (surely the model can generate zombie brain).

Generating test images with the test set encodings yielded the following (I randomly printed out 10 and below are 4 of those 10):

![image-20211101034715819](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101034715819.png)

As for the SSIM, it is as follows :

![image-20211101034755241](C:\Users\Sofia\AppData\Roaming\Typora\typora-user-images\image-20211101034755241.png)

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
