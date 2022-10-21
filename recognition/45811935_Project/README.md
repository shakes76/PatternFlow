# Generative Model of ADNI Brain Data Using VQ-VAE + PixelCNN

This project implements a generative model of the ADNI brain dataset, using a Vector-Quantised 
Variational Autoencoder (VQ-VAE), and a PixelCNN prior. Essentially, what this means is that a 
discrete, latent space can be learned by the VQ-VAE model, as a reduced representation of the ADNI 
brain dataset, from which, new discrete representations may be randomly generated, through the 
PixelCNN. These new representations may be decoded by the VQ-VAE again, to generate entirely new 
brain images.

## Algorithms/Models

### VQ-VAE

To understand VQ-VAEs, let us first review VAEs, i.e. Variational Autoencoders. Variational 
Autoencoders are an unsupervised deep learning model that learn a probability distribution over the 
latent space. The latent space refers to a reduced representation of the model inputs, from 
which, (ideally) near-identical reconstructions of the input may be formed.

The defining factor of a VQ-VAE, is that it essentially learns a discrete latent distribution 
instead.

This discrete representation is given by a collection of vectors known as a 'codebook', that 
are created via a Vector Quantization layer. These vectors are also known as embeddings, with 
the codebook/latent space also being known as an embedding space (see below).

![VQ-VAE Architecture](ReducedResults/VQVAEArchitecture.PNG)

### PixelCNN

...

## Dataset - ADNI Brain

## Training

## Validation - Hyperparameter Tuning & Model Selection

## Testing + Reconstructions

## Usage



The readme file should contain a description of the algorithm and the problem that it solves
(approximately a paragraph)

how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results,
if applicable.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation
and testing splits of the data

description and explanation of the working principles of the algorithm implemented and the problem it
solves (5 Marks)
2. description of usage


### Dependencies

- `tensorflow >= 2.10.0`
- `tensorflow-probability >= 0.18.0`
- `numpy >= 1.23.0`
- `matplotlib >= 3.5.2`
- `pandas >= 1.5.1`

To ensure correct dependencies are met, one may run the following command:

> pip install -r requirements.txt

## References
This project took influence from the original VQ-VAE paper, the PixelCNN Prior paper, and a 
VQ-VAE + PixelCNN Keras tutorial (Links below).

- https://arxiv.org/abs/1711.00937 - VQ-VAE Paper
- https://arxiv.org/abs/1601.06759v3 - PixelCNN Paper
- https://keras.io/examples/generative/vq_vae/ - Keras Tutorial

This is a very loose form a report. We dont need thorough analysis of results and experiments. When we mark your read me what we focus on  

how much work you have done,

what results you got, what what different model variants you have tried, what worked and what didnt 

did you understand the concept,

was your experiments reproducible given the instructions