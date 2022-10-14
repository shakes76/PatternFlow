# Synthetic Brain MRI Image Generation with VQ-VAE

## COMP3710 Pattern Flow Report

Alex Nicholson (45316207)

---
## Planning


Details:

* Model: [VQ-VAE (original)](https://arxiv.org/abs/1711.00937)
* Dataset: [OASIS](https://www.oasis-brains.org/#data)

Goals:

* “Reasonably clear image”
* [Structured Similarity (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity) index of over 0.6

---

## Project Overview

* **Description of the algorithm and the problem that it solves (approximately a paragraph)**
* **How it works (approximately a paragraph)**

---

## Setup

**List any dependencies required, including versions and address reproduciblility of results, if applicable.**

1. Install the conda envirinoment...
2. Download the OASIS dataset from [this link](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download)
3. Activate the conda environment
4. Run train.py

---

## Methods

**Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.**

---

## Example Generations

Below are some examples of the generations made by the VQ VQE model after 10 epochs of training over the full OASIS training dataset. These generations were produced by putting real MRI image examples from the test set into the model and then getting the reconstructed output from the model.

| ![alt text](./out/original_vs_reconstructed_0000.png)      | ![alt text](./out/original_vs_reconstructed_0001.png) |
| ----------- | ----------- |
| ![alt text](./out/original_vs_reconstructed_0002.png)      | ![alt text](./out/original_vs_reconstructed_0003.png)       |
| ![alt text](./out/original_vs_reconstructed_0004.png)      | ![alt text](./out/original_vs_reconstructed_0005.png)       |

## Generation Quality Over Time

Below is an animation of the progression of the quality of the model's generations over the course of training.
![alt text](./vqvae_training_progression.gif)

## Training Metrics

The various loss metrics of the model were recorded throughout training to track its performance over time, these include:

* Total Loss: What does the total loss represent???
* Reconstruction Loss: What does the reconstruction loss represent???
* VQ VAE Loss: What does the VQ VAE loss represent???

These losses are plotted over the course of the models training in both standard and log scales below:
![alt text](./training_loss_curves.png)

Model Log Loss Progress Throughout Training:
![alt text](./training_logloss_curves.png)

In addition to statistical losses, a more real world metric to track the quality of our generations over time is to compare the similarity of the reconstructed output images it produces with the original input image they are created from. This similarity can be measured by the SSIM (Structured Similarity Index). At the end of each epoch, the SSIM was computed for 10 randomly selected images from the test dataset, and the average was recorded. This average SSIM can be seen plotted over time below:
![alt text](./training_ssim_curve.png)

---

## TODO

- [x] Data importing
- [x] Model class
- [x] Model basic training function
- [x] Live training performance data logging
- [x] Output image results visualisation
- [x] Implement saving of output images to file
- [ ] Save images during training to show the progress
- [x] Training metrics over time plot
- [x] SSIM performance calculation
- [ ] Port my code over to the hpc for speedy slurm training
- [x] Do a big training run to push the standrard of output generations
- [x] Tune hyperparameters until results meet the standard
- [ ] Report writeup, etc.
