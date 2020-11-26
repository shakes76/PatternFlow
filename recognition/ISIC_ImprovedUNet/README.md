# Improved UNet for Image Segmentation of the ISIC 2018 Dataset

## The Dataset

The dataset is taken from the ISIC 2018 challenge task (https://challenge2018.isic-archive.com/). This dataset consists of 2,594 images of human skin taken via dermoscopy. An example image is shown below.

![image](https://github.com/Markopteryx/PatternFlow/blob/topic-recognition/recognition/ISIC_ImprovedUNet/images/image.png)

## Model

### Original UNet
The UNet is a neural network model originally developed in 2015 (https://arxiv.org/abs/1505.04597) for image segmentation. The model derives its name from the U-shaped network (shown below). Along the left path, the network is that of a fairly standard convolution neural network, consisting of two 3x3 convolutions, following by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.  After each downsampling the number of features channels are doubled and this is repeated until the bottom of the U. Along the right path, there are upsampling units followed by 2x2 convolutions alongside a concatentation with the corresponding result from the left side of the U (the grey arrows in the figure) (Ronneberger, Fischer and Brox, 2015).

![image](https://github.com/Markopteryx/PatternFlow/blob/topic-recognition/recognition/ISIC_ImprovedUNet/images/unet.png)

### Updated UNet
The updated UNet, and the basis of the network implemented here is the UNet by Isensee et al, 2017 (https://arxiv.org/pdf/1802.10508v1.pdf). The layout of this model is shown below and this is model used here.

![image](https://github.com/Markopteryx/PatternFlow/blob/topic-recognition/recognition/ISIC_ImprovedUNet/images/model.png)

Due to the nature of the task, the improved UNet was created for the specific dataset it was used on (brain image segmentation); therefore some changes had to be made for usage on the skin cancer dataset. First we only have two dimensional images here, so the dimensionality of the inputs is reduced by one. Additonally, we make use of the LeakyReLU activation function rather than the standard ReLU to prevent the dying ReLU problem seen when inputs of the ReLU are forced to zero. Other changes made for the improved UNet include removal of the max pooling layers in favor of downsampling convolutions, the addition of segmentation layers is used as a form of deep supervision (Isensee et al, 2017).

# Dependencies

## Driver Dependencies

* Python 3.7
* Tensorflow 2.3.1
* Scikit-Learn 0.23.2
* Numpy 1.18.5
* Matplotlib 3.2.0
* Glob
* OS

## Model Dependencies

* Python 3.7
* Tensorflow 2.3.1

## Usage

To use the code, only the driver script needs to be modified. Somewhere on the local machine two folders called 'data' and 'masks' should exist containing the images. The file paths in the driver script should be modified to point to this location.

## Result
An example result is shown below.

![image](https://github.com/Markopteryx/PatternFlow/blob/topic-recognition/recognition/ISIC_ImprovedUNet/images/predict.png)
