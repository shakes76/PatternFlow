# OASIS Brain Generative Model with DCGAN

---

> Using a deep convolutional adversarial generative model to generate images of MRI brain cross-sections from the OASIS Brain Dataset.

# Network Architecture

---

The adversarial generation model is based on the DCGAN architecture proposed by Radford et al. [RMC16] The implentation is detailed in Figure 1 and is as follows:
The generator network consists of 5 fractionally strided convolutions with stride height and width of 2 and a 3 x 3 kernel.
All convolutional layers have ReLU activations and are batch normalized except for the final convolutional layer which uses a Tanh activation, no batch normalization and has 1 filter equivalent to the number of channels of the brain image.

# OASIS Brain Generation Results

---

## Epochs 0 - 1200

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/generated_brains.gif)

# Real Brain Slices

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/real_brains.png)

# Generated Brain Slice at Epoch 1200

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/epoch_1200.png)

# Experimental Setup

Training was completed on the preprocessed training brain slices of size 256 x 256 and included 9664 samples. Additional preprocessing was completed to set the pixel values to between -1,1.
The generator network receives a noise variable as a 100-Dimensional vector sampled from a Gaussian distribution with mean=0 and std=1.
The network was trained using a Tesla V-100 GPU and took 12 hours to complete 1200 epochs with a batch size of 64.
