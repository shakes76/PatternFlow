Segment the 2018 ISICs dataset using Improved Unet 
======
Author: Yukun Zhan (45239502)

## Introduction
The ISICs data set comes from ISIC 2018 challenge data for skin cancer. The dataset contains 2594 damaged skin images with segmentation labels. The purpose of this project is to better classify and generate segmentation labels by constructing an improved unet model. The effect of the model will be evaluated by the dice coefficient.
## Improved Unet
The architecture of Improved Unet is similar to U-Net, both including a context aggregation pathway. The pathway encodes increasingly abstract representations of the input as it goes deeper into the network. The network also uses localization pathway, recombines these representations with shallower features to precisely localize the structures of interest. 

The authors of the network have written a number of integrated modules. 
### Context module
A context module consists of two 3x3 convolutional layers and a dropout layer (drop = 0.3), and connected by 3x3 convolutions with input stride 2. 
### Localization module
A localization module consists of a 3x3 convolution followed by a 1x1 convolution. 
### Upsampling module
A upsampling module consists of a 2x2 2D upsampling layer, and a 3x3 convolutional layer.

In addition, these modules use LeakyReLU activation layers. The specific network structure is shown in the following figure. 
![image](https://user-images.githubusercontent.com/79847033/150805413-e25a23de-05b6-453c-88ad-45777902b9d0.png)

## Dice coefficient
The Sørensen–Dice coefficient is a statistic used to gauge the similarity of two samples. The Dice Similarity Coefficient is a common metric used in segmentation problems. Formally, DSC is defined as: 
![image](https://user-images.githubusercontent.com/79847033/150813879-ed22e3f2-6719-41f2-baf7-cfcf80fb0dd6.png)
## Data preprocessing and spliting
Resize the image data and segmentation masks to (192, 256). The images data have 3 color channels, and the segmentation masks only have 1 color channels. Both the images and the segmentation masks are normalized. 

The dataset is divided into training set, validation set and test set in a ratio of 7:2:1.
## Prediction examples
The following figure is the comparison of the segmentation masks predicted by the model and the original labels
![image](https://user-images.githubusercontent.com/79847033/150814056-a30dba61-c078-43fa-bd07-06613a87eae9.png)

## Evaluation
The figure below is the dice coefficient values and loss values of the model in 20 epoch runs
![image](https://user-images.githubusercontent.com/79847033/150814174-589bb12a-e2c0-4671-aba5-b08048046212.png)
The figure below is the dice coefficient value and loss value of the model evaluated using the test dataset. The dice coefficient up to 0.8528
![image](https://user-images.githubusercontent.com/79847033/150814340-32525c02-3c40-454f-8ce7-55defaadea90.png)

## Dependencies
Python 3.9.5

Tensorflow 2.7.0

Tensorflow-addons 0.15.0

Numpy 1.22.1

Matplotlib 3.5.1
## References
Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-hein, K. H. (2018, February). Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. Retrieved from https://arxiv.org/abs/1802.10508

Wikipedia. (2022). Sørensen–Dice Coefficient. Retrieved from https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

Chandra, S. (2021). Report Pattern Recognition. Retrieved from comp3710 course page
