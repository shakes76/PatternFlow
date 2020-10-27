# Nic Thompson (s4436194) COMP3710 PatternFlow Project

This project seeks to implement a Deep Convolutional Generative Adversarial Network (DCGAN), for the OASIS brain 
dataset. This project specifically builds from scratch the generator and discriminator models required, 
and trains them on the corresponding data. Different resolutions have been implemented under this project, each varying
with the different model required for each. Namely, this project implements DCGAN models for,

* 28 by 28 images
* 64 by 64 images
* 128 by 128 images
* 256 by 256 images

The subsequent sections outline how to operate these scripts, and an explanation to the logic behind the implementation.

## Requirements

This project relies on various packages native to Python. Namely, this project requires at least,

* tensorflow 2.1.0
* matplotlib
* tqdm
* PIL
