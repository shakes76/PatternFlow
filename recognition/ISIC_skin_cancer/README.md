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

The filenames of the mask images should have the id which is same as the filenames of the RGB images with "_segmentation" ending.
The size of the mask image for the corresponding RGB image should be same.
Even our preprocessed dataset have 2,594 train images and its corresponding segmentation mask, which is quite large dataset.

and you will see that during the training phase, data is generated in parallel by the CPU and then directly fed to the GPU.
For more detailed reference of implementation of data generation part, check this tutorial:
[A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
