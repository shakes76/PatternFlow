# Super-Resolution Network for 2D Brain MRI

Task 5 Description: Implement a brain MRI super-resolution network by training on the ADNI brain dataset. Create down-sampled data(approximately by a factor of 4) using Tensorflow implementations. The network should be trained to up-scale from 4x down-sampled input and produce a “reasonably clear image".

## 1. Introduction

The target model is able to reconstruct a 4-scale high-resolution image from a given low-resolution image. The principle is to train a convolutional neural network with pairs of low-res images and high-res images and optimize the mean squared error between prediction and high-res images. The peak signal-to-noise ratio (PSNR) is the main metric to quantify and evaluate the quality of reconstructed images. The basic model was inspired by [Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/#define-utility-functions).

### 2. Running Environment

Requirements
- Python 3.9
- Tensorflow 2.9.2
- numpy, os, math, matplotlib, PIL

## 3. Training

### 3.1 Dataset

In this project, I train the model on the [ADNI MRI Dataset](https://adni.loni.usc.edu/). In particular, I download the pre-processed training set and validation set from the course Blackboard site under Course Help/Resources/[ADNI](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download). I just use the images in the AD train as my whole dataset. The number of images is 10400 and it is split into two parts: 9360 training examples, and 1040 validation examples. I use images in AD_NC/test/AD as test images for evaluation at the end.

### 3.2 Training

The model is showed below

    def get_model(upscale_factor, channels):
        conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        inputs = keras.Input(shape=(None, None, channels))
        x = layers.Conv2D(64, 5, **conv_args)(inputs)
        x = layers.Conv2D(64, 3, **conv_args)(x)
        x = layers.Conv2D(32, 3, **conv_args)(x)
        x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
        outputs = tf.nn.depth_to_space(x, upscale_factor)

        return keras.Model(inputs, outputs)

I use [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) with the default setting as the optimizer and [MeanSquaredError](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError) as the loss function. Set epoch to 100, calculating the mean squared error between prediction and high-res images then adding back-propagation. In the training step, the PSNR is close to 33.

### 3.3 Important Modules

#### 3.3.1 dataset

Define functions to load data from given path, resize images and assign in batch, devide training set and validation set, nomalization.

#### 3.3.2 modules

Define functions to get model, down-sampling, up-scaling, plot result images.

#### 3.3.3 train

Define functions for calculating PSNR, ModelCheckpoint and EarlyStopping callbacks.

There are several parameters that need to be decided:
- crop_size: Original image size is 256 x 240. After many times tests, I resize it to 300 x 300 for better performance.
- upscale_factor: In this case, it is 4.
- batch_size: I decide to use 20 as batch size. Because in the dataset, 20 images are as one group, too small or too large a batch size will lead to bad results.
- epoch: I set epoch to 50 or 100.

#### 3.3.4 predict

There is no function and parameters.

## 4. Prediction

PSNR is the main metric to quantify and evaluate the quality of reconstructed images.

![An example result image with input, target and prediction:](https://github.com/WYF0001/PatternFlow/blob/topic-recognition/recognition/s45533675/results.png)

PSNR of low resolution image and high resolution image is 25.7794.
PSNR of predict and high resolution is 27.4004.
