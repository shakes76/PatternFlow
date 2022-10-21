# OASIS Brain Generative Model with DCGAN

---

> Recognition Problem  
> Use a deep convolutional adversarial generative model to generate realistic 256x256 grayscale images of brain MRI cross-sections from the OASIS Brain Dataset [2].

# DCGAN Implementation

---

The adversarial generation model is based on the DCGAN architecture proposed by Radford et al. [1] The implementation is detailed in Figures 1 and 2 is as follows:  
The generator network receives a noise variable as a 100-Dimensional vector sampled from a Gaussian distribution with mean=0 and std=1, at no stage is the generator exposed to images from the training dataset, instead the generator learns its weights using the gradient of the discriminator.  
The generator network consists of 5 fractionally strided convolutions with stride height and width of 2 and a 3 x 3 undilated kernel, weights are initaliased with the Glorot normal initializer.All convolutional layers have ReLU activations and are batch normalized except for the final convolutional layer which uses a Tanh activation, no batch normalization and has 1 filter equivalent to the number of channels of the brain image.  
The discriminator network similar to the original implementation uses convolutional layers with LeakyReLU activations of slope 0.2.  
The Adam optimizer was used with a learning rate of 0.0002 for the generator and 0.0001 for the discriminator with default beta_1=0.9 momentum terms, both networks were updated using cross entropy loss.  
Checkerboard-artifacts are seen in the early stages of training and common in fractional strided convolutions[3], in nearly all validation runs the generator was able to overcome this by epoch 100.

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/gen_arc.png)

###### Figure 1: Generator Network

###

###

###

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/disc_arc.png)

###### Figure 2: Discriminator Network

###

###

###

# Dependencies

---

- Matplotlib 3.2+
- Tensorflow version 2.2.0+ (Keras included)
- Python 3.6.9

# Usage

---

#### To train and test the model:

```python
$ python test_dcgan.py
```

#### Directory Structure

Download the OASIS dataset and unzip into working directory:

```
test_dcgan.py
layers.py
losses.py
keras_png_slices_data/
    keras_png_slices_train/
        *.png
```

# Examples

---

### Epochs 0 - 1200

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/generated_brains.gif)

### Real Brain Slices

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/real_brains.png)

### Generated Brain Slices

![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/epoch_1200.png)
![alt text](https://storage.googleapis.com/dl-visuals-peter-ngo/disc_acc%20and%20ce.png)

# Training and Datasets

The network was trained using a virtual instance of a Tesla V-100 GPU and 32GB Ram, training took 8 hours to complete 1200 epochs with a batch size of 64.
No training and test splits were used as the objective of the recognition problem was to generate photorealistic outputs as determined by a human observer.  
Training was completed on a preprocessed version of the OASIS brain dataset [2] which provided training brain slices of size 256 x 256px containing 9664 samples and accounting for 85% of the total samples, with 32 slices per subject and 302 subjects.The heterogeneity of the brain slices also meant that assigning labels to specific cross-sections was not possible.  
The tuning of hyperparameters such as filter size, kernel size and stride, batch-size and dilation, learning rates was completed on the validation dataset with 1120 samples, the purpose of which was to infer a good approximation of a suitable range for these values when completing the training and testing on the main 9664-sample dataset. Examples included adjusting the batch-size to 64 in order to avoid memory errors and testing dilated convolutions for denoising performance.

# References

---

[1] A. Radford, L. Metz, and S. Chintala, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks,” arXiv:1511.06434 [cs], Jan. 2016, arXiv: 1511.06434. [Online]. Available: http://arxiv.org/abs/1511.06434  
[2] OASIS Brain Dataset: https://www.oasis-brains.org/#access  
[3] Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.0000
