# DCGAN for OASIS dataset

Tensorflow implementation of Deep Convolutional Generative Adversarial Networks that solves COMP3710 Report Problem 6 (OASIS brain dataset).
![Image of DCGAN](https://gluon.mxnet.io/_images/dcgan.png)
![Image of OASIS brain dataset](sample_images/Original_Images.png)
## Prerequisites
* Python 3.7
* Tensorflow 2.1
* Matplotlib

## Usage
### Dataset
The OASIS preprocess dataset can be downloaded [here](https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ) and need to be put inside the main folder to train the network.
### Train
To train the DCGAN network, run

    python main.py


## Results
After 10th epoch:
![Image of 10th epoch](sample_images/image_at_epoch_0010.png)
After 50th epoch:
![Image of 50th epoch](sample_images/image_at_epoch_0050.png)
After 80th epoch:
![Image of 80th epoch](sample_images/image_at_epoch_0080.png)
After training:
![Image of after training](sample_images/Examples.png)

### Loss Curve
Losses of Generator and Discriminator.
![Image of loss curve](sample_images/Loss_Curve.png)

## Author
Ganze Zheng(44570776)
