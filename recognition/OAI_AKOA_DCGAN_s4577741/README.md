# Recognition using DCGAN with OAI AKOA knee data set
##### Richard Roth - s4577741

## Algorithm Description
The algorithm is a DCGAN (Deep Convolution Generative Adversarial Network) made up of a generator, discriminator and driver script. A GAN (Generative Adversarial Network) contains
two models which are trained simultaneously by an adversarial process to improve each others performance in their respective tasks, generating an image from noise and 
discriminating between real and generated images. During training, the generator uses noise to progressively generate more accurate looking images, while the discriminator 
becomes increasingly capable of telling the real images from the generated images apart. The process converges toward an equilibrium where the discriminator can no longer distinguish real images from generated images.

![Image](https://www.tensorflow.org/tutorials/generative/images/gan1.png)
![Image](https://www.tensorflow.org/tutorials/generative/images/gan2.png)

## Problem Description
The OAI (OsteoArthritis Intiative) AKOA (Accelerated Knee OsteoArthritis) data set contains 18,000 images of longitudinal MRI (Machine Resonance Imaging) sections from over 500 patients. These patients have symptoms of osteoarthritis, which are represented by features in the MRI images, which experts can determine through training and experience. Using human experts to analyse and determine features of osteoarthritis is expensive in labour and time and would benefit from machine automation. 

![image](https://vetmed.tamu.edu/news/wp-content/uploads/sites/9/2015/03/levine-scan.jpg)

## Solution Description
DCGANs are adept at learning how to recognise features and can be trained to generate faces, animals and objects. If trained on a set of osteoarthritis MRI images, a DCGAN can learn the features of osteoarthritis such as joint space narrowing, sclerosis, osteophytosis, joint erosion, subchondral cysts, bone marrow lesions and synovitis. Once the discriminator GAN has learned these features it can be turned into a model. The model can then be given MRI images of a knee and determine whether that knee is healthy or shows symptoms of osteoarthritis. 

![image](https://www.researchgate.net/publication/285383083/figure/fig6/AS:614251773296648@1523460443185/MRI-of-markers-of-inflammation-in-OA-Fluid-sensitive-sequences-are-capable-of.png)

## How it works
The generator uses random noise to generate images and when the discriminator either chooses the real image or makes an error in identifying the real image, the discriminator uses that information to iterate on the generated noise, moving closer to a real image the discriminator can't tell apart. Both the generator and descriminator are CNNs (Convolutional Neural Networks) with different structures for their respective tasks. The generator uses noise which is transposed over successive convolutional layers to form the final image. Conversely, the discriminator uses the transposed image of the generator and applies convolutions to deconstruct the image into recognisable features. Generally convolutional layers are grouped into blocks, those layers containing sub-layers of perceptrons, which are a mathematical model for biological neurons. As the model is trained, weights adjust in each of these perceptrons to provide meaning to the sub-layers, which pass this along to the layers and finally the blocks, which in turn create the model for the individual CNNs, which contribute to the overall DCGAN.

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

Initially I used the same generator and discriminator structure I used on the CelebA data set which lead to poor results. This was using Selu, Nadam and He_Uniform.

![image](https://drive.google.com/uc?export=view&id=1-2liqiT0jfqhMm0yQhwLaZpaieyo42hv)
![image](https://drive.google.com/uc?export=view&id=1-4hBMtfpumErGUNxgxfsw6MFcw2QeoD7)

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

Couldn't upload the gif as it was 120MB, but here is the training history. This was an optimised model using ReLU and Adam. 

![image](https://drive.google.com/uc?export=view&id=15pEJsZ_6Hk-MDYx-NXOpbdfZzmuknpLY)
https://drive.google.com/file/d/15Lb3ex_bsfsvk78bVFgf11r0hbJqjhV5/view?usp=sharing

## Test 8 & 9

I started experimenting with selu and nadam using an optimised model with varying results. 

![image](https://drive.google.com/uc?export=view&id=1tp01gRCpxea6o-EasUPyquQ6ex_UrkfQ)
https://drive.google.com/file/d/1F_Gl6UwV22Uyprh5EdrZYm9vMHYYp-17/view?usp=sharing

## Test 10

The optimised model was showing improving results and I started to see good progress with SeLU & Adam.

![image](https://drive.google.com/uc?export=view&id=1NQQfw6sdXRM88YlioQz2Gqvt4Cgggg6q)

## Test 11

Elu/Nadam - not much to say.

![image](https://drive.google.com/uc?export=view&id=1-AvQr6HC-DVp5nFEpJu51gOudYORWQqI)

## Test 12

I started going back to LeakyReLU and Adam and tried 10000 epochs with batch sizes of 6.

![image](https://drive.google.com/uc?export=view&id=1NayGXQGmwHj4fgw6XmIP1azfG_YWS4Ho)
https://drive.google.com/file/d/15RO99sIhP84K6Rh-w-Hme0KqG6ZRv-nj/view?usp=sharing

## Test 13

Wanted to try the optimised model on Elu & Nadam and the results were promising. 

![image](https://drive.google.com/uc?export=view&id=1pQwtwfO3yS0FTbY4xiBdOYadfu950Dx7)
https://drive.google.com/file/d/1BeQ-2xWjS2ll_abTrEzPnJApojtonPaB/view?usp=sharing

## Test 14

The final test was done with a much larger neural network which achieved impressive results despite only doing 1000 epochs on a batch size of 6. I don't really understand why the G and D loss are all over the place, but the results were good, so that maybe requires some investigation. 

![image](https://drive.google.com/uc?export=view&id=15JJ3-B0siarlFZQRi7WAHOlu6LCi_0hz)
![image](https://drive.google.com/uc?export=view&id=1cgiVTgquLNAseAaxSn7RXfWqezU51Q-L)
https://drive.google.com/file/d/15TC7NCXRZIksBJoA2-eoINyIO0AxIbSU/view?usp=sharing

## Justification of Training, Validation and Testing split of data

DCGANs do not have complex data preprocessing. I was thinking about experimenting with segmenting the data sets, but that would detract from the power of GANs and their ability to discern features from massive amounts of information. 

Maximum training set used in my model were 7200 images and 1440 images for the testing set. There were memory limitations on using more images and I wonder if adding more would have improved the model. 


