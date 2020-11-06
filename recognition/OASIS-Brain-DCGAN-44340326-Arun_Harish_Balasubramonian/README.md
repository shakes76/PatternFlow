# Deep Generative Adversarial Network (DCGAN)
_Author_ : _Arun Harish Balasubramonian_

_Student No_ : _44340326_

_Dataset_ : _OASIS Brain_

_Email_ : s4434032@student.uq.edu.au _or_ a.balasubramonian@uq.net.au

## Description GAN
GAN model encompasses two neural networks called the Generator, and the Discriminator. Both of these models compete with each other, having the intention to fool their opposition. The generator has the intention to produce images that would trick the discriminator in believing it has real. 

The discriminator being trained with real dataset would compete in identifying the fake images produced by the generator. The model is said to have succeded if both the model reach an equilibrium with 50% of the generator dataset being identified as fake by the discriminator: the image produced by the generator now has similar feature to the original dataset. 

The model works by inputting random noise dataset to the generator, that __upsamples__ them using transposed convolutional to produce images. These images are evaluated by the discriminator by __downsampling__ the input dataset updating its internal weights. and its output

## Description DCGAN
DCGAN is a variant of GAN, but with slightly different design on the neural network. According to the paper : [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf), there are certain heuristics on the Deep learning models of GAN that gives optimal / stable result, suggesting the following architecture:
* Replacing the us
* Using Batch Normalisation on both generator and discriminator neural networks.
* Using ReLU activation for all generator layers except for the output layer using Tanh
* Using LeakReLU activation for all discriminator layers.

## Test Script

## Result
The entire image dataset in `keras_png_slices_train` was used for 20 epochs yielding the following results:
