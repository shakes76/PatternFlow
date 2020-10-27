# Image Segmentation Keras : Implementation of Segmentation on the ISICs data set with the [Improved UNet](https://arxiv.org/abs/1802.10508v1)
I used Keras with a Tensorflow backend. This UNet was built for the pre-processed version of ISICs data set (as provided by COMP3710 teaching team).  
However, if you are intereted in to download the original dataset, it is on: https://challenge2018.isic-archive.com/

## Models
Example results for the pre-trained models provided :

Input Image            |  Original Segmentation Image|  Output Segmentation Image
:-------------------------:|:-------------------------:|:-------------------------:
![](resources/x_train_11.png)  |  ![](resources/y_train_11.png)|  ![](resources/preds_train_11.png)
![](resources/x_train_12.png)  |  ![](resources/y_train_12.png)|  ![](resources/preds_train_12.png)

## Getting Started

### Prerequisites

* Keras 2.4.3
* Tensorflow 2.3.0
* Numpy 1.18.5 
* Matplotlib 3.2.2
* Skimage 0.16.2

if you have not already install any package, you can use command 
```shell
pip install [package name]
```

### Preparing the data for training

You need to make two folders

* Images Folder - For all the training images
* Masks Folder - For the corresponding ground truth segmentation images

after you download the dataset, you can use command 
```shell
mkdir dataset
cd dataset
mkdir images
mkdir masks
cd folder_contain_train_images
cp * dataset/images
cd folder_contain_mask_images
cp * dataset/masks
```
Then you are good to load the dataset images!

### Data Generator
The filenames of the mask images should have the id which is same as the filenames of the RGB images with "_segmentation" ending.
The size of the mask image for the corresponding RGB image should be same.
Even our preprocessed dataset have 2,594 train images and its corresponding segmentation mask, which is quite large dataset.

and you will see that during the training phase, data is generated in parallel by the CPU and then directly fed to the GPU.
For more detailed reference of implementation of data generation part, check this tutorial:
[A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

## Improved Unet Structure
[](resources/Improved_UNET.png)

## Evaluation Metrics

Predicted responses are scored using a threshold [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) metric.

Image segmentation is basically the same as classification problem which means trying to determine the class of pixels. Given an image, we want to detect the scar on the skin. The problem is that usually the image is very high-dimentional, it has like around ten thousand pixels and we want to classify each pixel in the image whether it is a background pixel or if it is a foreground pixel - the pixels belonging to the class we are trying to detect.  
By nature, usually these problems are highly unbalanced, so these classification problems are unbalanced because most of the pixels are usually the background pixels, usually around 90% of them, so the remaining 10% could be e.g. 10 different classes if you are trying to segment.  
Therefore, if the datasets or the images are unbalanced, the pixel wise accuracy which we usually use to evaluate classification problems will no longer be valid or proper matrix to use. The reason is that if you for example predict every pixels as background then your accurary will already be 90% which is too high, it is biased. So what we want to do is instead use matrix that specifically target the pixels in the foreground classes.  
The Sorensen-Dice coefficient is used as a similairty metric and is commonly used as a metric for segmentation algorithms.  
The original formula was intended to be applied to binary data. Given two sets, X and Y, it is defined as 2|X||Y|/|X|+|Y|, where |X| and |Y| are the cardinalities of the two sets.
