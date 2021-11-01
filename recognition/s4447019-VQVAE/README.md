# Implementation of VQVAE for COMP3710 Project

## Algorithm description
1. The read me file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.

![caption](/recognition/s4447019-VQVAE/images/vqvae.model.png)
Figure 1. Diagram of model from paper (model from Neural Discrete Representation Learning,  2018).

## Dependencies
The list of dependencies required for this implementation are as follows:

* python 3
* tensorflow 2.6.0
* tensorflow-probability 0.14.0
* numpy 1.21.3
* matplotlib 3.4.3
* scikit-image 0.18.1

## Example outputs & plots
3. provide example outputs and plots of your algorithm code

![caption](/recognition/s4447019-VQVAE/images/results1.png)
Figure 2. Original and Reconstructed results

![caption](/recognition/s4447019-VQVAE/images/ssimcsa.png)
Figure 3. SSIM

![caption](/recognition/s4447019-VQVAE/images/results2.png)
Figure 4. Original and Code results

## Justification of training, validation and testing split of data
5. describe and justify your training, validation and testing split of the data.

The spilt between the datasets was 90% for training, 10% for validation and 5% for testing. Due to the large number of usable images, testing only requires a 5% share of the total images. It is better off using as many images as possible for the training.