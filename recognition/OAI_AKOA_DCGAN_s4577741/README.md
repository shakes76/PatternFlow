# Recognition using DCGAN with OAI AKOA knee data set
##### Richard Roth - s4577741

## Algorithm Description
The algorithm is a DCGAN (Deep Convolution Generative Adversarial Network) made up of a generator, discriminator and driver script. A GAN (Generative Adversarial Network) contains two models which are trained simultaneously by an adversarial process to improve each others performance in their respective tasks, generating an image from noise and discriminating between real and generated images. During training, the generator uses noise to progressively generate more accurate looking images, while the discriminator becomes increasingly capable of telling the real images from the generated images apart. The process converges toward an equilibrium where the discriminator can no longer distinguish real images from generated images.

![Image](https://www.tensorflow.org/tutorials/generative/images/gan1.png)
![Image](https://www.tensorflow.org/tutorials/generative/images/gan2.png)

## Problem Description
The OAI (OsteoArthritis Intiative) AKOA (Accelerated Knee OsteoArthritis) data set contains 18,000 images of longitudinal MRI (Machine Resonance Imaging) sections from over 500 patients. These patients have symptoms of osteoarthritis, which are represented by features in the MRI images, which experts can determine through training and experience. Using human experts to analyse and determine features of osteoarthritis is expensive in labour and time and would benefit from machine automation. 

![image](https://vetmed.tamu.edu/news/wp-content/uploads/sites/9/2015/03/levine-scan.jpg)

## Solution Description
DCGANs are adept at learning how to recognise features and can be trained to generate and distinguish faces, animals and objects. If trained on a set of osteoarthritis MRI images, a DCGAN can learn the features of osteoarthritis such as joint space narrowing, sclerosis, osteophytosis, joint erosion, subchondral cysts, bone marrow lesions and synovitis. Once the discriminator GAN has learned these features it can be turned into a model. The model can then be given MRI images of a knee and determine whether that knee is healthy or shows symptoms of osteoarthritis or even be capable of identifying which specific features of osteoarthritis the patient is exhibiting in the MRI.

![image](https://www.researchgate.net/publication/285383083/figure/fig6/AS:614251773296648@1523460443185/MRI-of-markers-of-inflammation-in-OA-Fluid-sensitive-sequences-are-capable-of.png)

## How it works
The generator uses random noise to generate images and when the discriminator either chooses the real image or makes an error in identifying the real image, the discriminator uses that information to iterate on the generated noise, moving closer to a real image such that the discriminator can't tell them apart. Both the generator and descriminator are CNNs (Convolutional Neural Networks) with different structures for their respective tasks. The generator uses noise which is transposed over successive convolutional layers to form the final image. Conversely, the discriminator uses the transposed image of the generator and applies convolutions to deconstruct the image into recognisable features. Generally convolutional layers are grouped into blocks, those layers containing sub-layers of perceptrons, which are a mathematical model for biological neurons. As the model is trained, weights adjust in each of these perceptrons to provide meaning to the sub-layers, which pass this along to the layers and finally the blocks, which in turn create the model for the individual CNNs, which contribute to the overall DCGAN.

![image](https://gluon.mxnet.io/_images/dcgan.png)

## Dependencies

* Sys
* Numpy
* Os, Time
* Matplotlib
* Random
* Keras
* TQDM
* Tensorflow
* Google.colab

## Methodology

### Test 1

From previous work on the CelebA dataset, a dataset containing images of celebrities, I utilised the same model for the generator and discriminator as a baseline test as I knew I had generated good results using it on discerning features of a face. For the CelebA dataset I had settled on models that utilised a block structure with the SeLU (Scaled Exponential Linear Unit) activation function, the Nadam (Adam with Nesterov Momentum) loss function and the He_Uniform (Uniform Distribution) kernel initialisation. Some results of this work are below. 

![image](https://drive.google.com/uc?export=view&id=1f4-FQjW62d4CZxU5ZdNasLekQ_oKQUEY)
![image](https://drive.google.com/uc?export=view&id=1-pYOEJIffi0bFQGVwwsHNXO4oR4bVS-7)

During this baseline test for the AKOA knee data I changed the the shape of the image to (X,Y,1) to account for greyscale images and modified the models to work with an image shape of (260, 228, 1). Initial tests were done using Selu, Nadam and He_Uniform with epochs between 3000 and 6000 using batches of 48.

![image](https://drive.google.com/uc?export=view&id=1-2liqiT0jfqhMm0yQhwLaZpaieyo42hv)
![image](https://drive.google.com/uc?export=view&id=1-4hBMtfpumErGUNxgxfsw6MFcw2QeoD7)

Testing revealed heavy pixelation on the generated images and poor performance by the discriminator. OOM (Out of Memory) errors were also an issue due to the large batch size.

### Test 2

I started experimenting with parameters and using ReLU and Adam for better results, but there was still an issue with my generator. Also, the image tended to diverge into obscurity.

![image](https://drive.google.com/uc?export=view&id=10Aoh2gO1p64cLYBFV5dkHGppSp5l6JeX)
![image](https://drive.google.com/uc?export=view&id=15K7IAousLkYVqYib1MzyJFTBF5by3v7j)

## Test 3

I thought it may have to do with my kernel intialiser so I tried glorot_uniform to poor results. 

![image](https://drive.google.com/uc?export=view&id=1BoHGXFzJhDiqdPfhDtQsdqPKDV_olJlC)
![image](https://drive.google.com/uc?export=view&id=1H5dB2Fd5aNCTIV03NWI_e-6TcFghSh_f)

## Test 4

Tweaked some of the parameters for better results, but was still seeing the pixelation which I suspected was a generator problem.

![image](https://drive.google.com/uc?export=view&id=15EWpqop-M048QBCOxQIzpMV3VyH3r8TI)
![image](https://drive.google.com/uc?export=view&id=1C8JnE3cvLQ22RlB6vQRidpJb2LMAwNZS)

## Test 5

I was still adament on making SeLU/Nadam work, but the results were not improving greatly. 

![image](https://drive.google.com/uc?export=view&id=1EuAVSbIekSZZO998z9y47in0v7f9OKvE)
![image](https://drive.google.com/uc?export=view&id=1IHLmUH8BPU0D8dLT-DKEnctAx6MCAETb)

## Test 6

Test 1 to 5 I was using a larger training and validation set, and thought that as the variation in images is quite large across the set, I wondered what would happen if I made the training and testing set much smaller.

![image](https://drive.google.com/uc?export=view&id=1-j79RDoYROtrbKrR1U720_MYPBaxSjG4)
![image](https://drive.google.com/uc?export=view&id=15KO2dSg4T31X9u4eJHhvZd5D8PdnTF4W)

## Test 7

Gif shows the performance of the model. This was an optimised model using ReLU and Adam. 

![Alt Text](https://drive.google.com/uc?export=view&id=1hc5vO8ceG6n5764-Qq9-EGQCIHFraXQM)
![image](https://drive.google.com/uc?export=view&id=15pEJsZ_6Hk-MDYx-NXOpbdfZzmuknpLY)

## Test 8 & 9

I started experimenting with selu and nadam using an optimised model with varying results. 

![Alt Text](https://drive.google.com/uc?export=view&id=1LZFKkXO6qFcXltZSa5fsmFffq_WNbi5c)
![image](https://drive.google.com/uc?export=view&id=1tp01gRCpxea6o-EasUPyquQ6ex_UrkfQ)

## Test 10

The optimised model was showing improving results and I started to see good progress with SeLU & Adam.

![image](https://drive.google.com/uc?export=view&id=1NQQfw6sdXRM88YlioQz2Gqvt4Cgggg6q)

## Test 11

Elu/Nadam. Training failed due to connection issues, model was abandoned.

![image](https://drive.google.com/uc?export=view&id=1-AvQr6HC-DVp5nFEpJu51gOudYORWQqI)

## Test 12

I started going back to LeakyReLU and Adam and tried 10000 epochs with batch sizes of 6.

![Alt Text](https://drive.google.com/uc?export=view&id=1Cz3Ua14TL7BtmVg9lpX3KUc8mBsHJKFk)
![image](https://drive.google.com/uc?export=view&id=1NayGXQGmwHj4fgw6XmIP1azfG_YWS4Ho)

## Test 13

Wanted to try the optimised model on Elu & Nadam and the results were promising. 

![Alt Text](https://drive.google.com/uc?export=view&id=1p7FpFl2taJUR5p3qvfP4zp9ciDdwIO2g)
![image](https://drive.google.com/uc?export=view&id=1pQwtwfO3yS0FTbY4xiBdOYadfu950Dx7)

## Test 14

The final test was done with a much larger neural network which achieved impressive results despite only doing 1000 epochs on a batch size of 6. I don't really understand why the G and D loss are all over the place, but the results were good, so that maybe requires some investigation. 

![Alt Text](https://drive.google.com/uc?export=view&id=11Q0NuVVQxjs-VqGL26R5aelXXJRiiFRu)
![image](https://drive.google.com/uc?export=view&id=15JJ3-B0siarlFZQRi7WAHOlu6LCi_0hz)
![image](https://drive.google.com/uc?export=view&id=1cgiVTgquLNAseAaxSn7RXfWqezU51Q-L)

## Test 15

Highly experimental test with extremely optimised model. Attempted to integrate SSIM, but was unsuccessful. Model contains less than 90,000 parameters on the generator and less than 2mil on the discriminator allowing for excessively large batch sizes. SSIM was not working as intended, but images looked good.

![image](https://drive.google.com/uc?export=view&id=110ixJY6VfDFB5GhCQlXwxFI0qyVy671C)
![image](https://drive.google.com/uc?export=view&id=135vQrnbsbYP1Q_aG_K8mi2Nzs6-EAjcG)
![image](https://drive.google.com/uc?export=view&id=149d7LbrLSqwoTVZJc2M7s_AbOf9nHfiY)

## Justification of Training, Validation and Testing split of data

An initial experiment was attempted on all 18000 images of the OAI AKOA dataset, but the memory requirements were too great to process. Subsequent attempts were made on 1000, 1200, 2400, 3600 and 7200 images (between 6% and 40% of the data). The testing data was set at 20% of training data as per standard practice. 

Testing showed that dataset size was proportional to batch size in terms of results. A small training set with a small batch size led to poor results, but a small training set with large batch size showed similar results to a large training set with small batch size. This is due to the nature of DCGANs as they discriminate the features in the data they are given. Larger batch sizes means more training in less time, so it can be posited that more epochs on a small batch size with small training set could lead to similar results, but this would require more in-depth testing. 

Caveat, a DCGAN is only as good as its data and a well trained GAN would require patients exhibiting all features of Osteoarthritis to properly learn the symptoms to distinguish them. When evaluating a model this is an important consideration as the final test (test 15) shows good results but was only done with a training set of 1000 images. However, as pointed out, Test 15 is highly optimised and was not tested on the full set of 18000 images, which may now be possible due to the high level of optimisation.

In conclusion, the effectiveness of a DCGAN is relative to training dataset, but evaluation and testing can be managed on smaller datasets to save on time and resources. If a DCGAN is capable of generating good results with a small dataset, then it will also generate good results with a larger dataset. 


