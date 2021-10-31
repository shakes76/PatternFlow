# Segment the ISICs data set with the Improved UNet

This project is aimed to train a improved UNet model for detecting the melanoma shape from its photo. After properly trained, the model achieves the accuracy over 0.92 and dice similarity coefficient (DSC) over 0.8.

## BACKGROUND
The International Skin Imaging Collaboration is an internationa effrort to improve melanoma disgnosis. The project target is to develop image analysis tools to enable the auto mate disgnosis of melanoma from dermoscopic images [1]. The challenge is divided into three sub tasks which are:
Task 1: Lesion Segmentation 
Task 2: Lesion Attribute Detection 
Task 3: Disease Classification 

In our project, only task one is focused and addressed. Our model use Imporved UNet to segement the lesion area from background with a high accuracy and DSC. 

## DATASET DESCRIPTION

The datasets used in this project from training, validating and testing are all from ISICs 2018 Challenge Data set. The datasets contains two matched groups with each group has 2594 images. The first group of images is the raw melanoma images and the second groups of image (mask image) is the segmentation of the lesion area which will serve as labels in following model training.  mask images are binary images with only pixel value of 0 and 255. Pixel value = 255 means the lesion area while 0 means background area. Figure 1 is an example of melanoma raw image (group 1 image) and Figure 2 is the responding mask image.

![](./images/ISIC_0000008.jpg)

Figure 1, An example of melanoma raw image

![](./images/ISIC_0000008_segmentation.png)

Figure 2, Mask image (segmentation image) of Figure 1 
## DATA PREPROCESSING

Data preprocessing of this project including three steps:

### 1. Load Rwa Data

In this project, the grayscale version of images are loaded for training in the model because the task of this project is to segment the lesion area without analysising the attribute of lesion area. Load the grayscale only can be helpful in reducing the unnecessary computing resources. 

### 2. Split Raw Data

There are 2594 pairs of image & mask supplied. In this project, 60% of them will be used for model training and 20% of them for model calidation and the rest 20% for testing the trained model. That is, training set has 1556 pairs of images, both validating and testing sets have 519 pairs of images. To make sure the credibility of the model, all datasets will be shuffled before training. 

### 3. Map and Decode Raw Images

Before datasets are ready to be trained in the model, all images and masks will be rescaled into the same size with matching mask. In this model, the predefined image size is 256 * 192 because the balance between image actual size and computing resources. The raw image size distribution is as following. Since the number of large image accounts for over 2/3, it is not suitable to rescale image to a very small size, but large size will cause slow training process. 256 * 192 is the chosen predefined image size. 512 * 384 is also trained but donot get a pleasant result (DSC is only over 0.7 not 0.8). Figure 4 is the example of resized grayscale image and its mask. Size of both image and mask are 256 * 192.


![](./images/width_distribution.png)

Figure 3, width distribution of all images

![](./images/resized.png)

Figure 4, Example of resized image and mask


## MODEL TRAINING 

### Model Description
U-Net is a convolutional neural network that is originally developed for biomedical image segmentation [2]. It has two stage. The first stage is VGG stage to encode the image and the second stage is CONCAT stage by combining the symmetric VGG with previous step out put for further convolution. Figure 5 is a sampled structure of UNet. 

![](./images/unet.jpg)

Figure 5, Sample Unet Structure

Each VGG stage contains a seriers of convolution layer with a max pooling layer to encode raw image. CONCAT stage requires up sampling (upsampling with convolution) at first and concat output of symmetric VGG concolution output. then perform a seriers of convolution action and also perform a max pooling action. Final convolution layer will be used to segment the image based on the predefined settings (num of classes, activation functions and so on). 


### Model Structure

Model straucture of this project is as following with four VGG block, one bottom block and four CONCAT block. The output layer use 'Sigmoid' as the activation function.

Input size for each image: (256， 192， 1)

Encode output size for each image: (16, 12, 1024)

Figure 6 shows the structure of applied UNet.

![](./images/UNet_cons_1.png)

![](./images/UNet_cons_2.png)

Figure 6, UNet structure in this project

## Training History

In this project, 100 epochs of training are applied with the bach size of 8. 
### Training Result at the First Epoch
![](./images/epoch1.png)
### Training Result at the 50th Epoch
![](./images/epoch50.png)
### Training Result at the Final Epoch
![](./images/epoch100.png)

### Training Accuracy vs Validation Accuracy
![](./images/accuracy.png)
### Training Dice Coefficient vs Validation Dice Coefficient
![](./images/DSC.png)

### Loss of Trainig Dataset vs Loss of Validating Dataset 
![](./images/loss.png)

## RESULTS
As can be seen from the training history, accurcay (Acc) above 90% and dice coefficient (DSC) above 0.8 are achieved in the model. The Acc and DSC of validating dataset began over 90% and 0.8 at 13th epoch and 45th epoch respctively.

### Model Evaluation Display

The average DSC of 519 test data achieves up to 0.8179469 which is higher than preset 0.8. 

The following four pictures show four testing results from testing dataset. In each row, first image is the grayscale raw image and the second image is the mask and the last one is the predicted mask. The model succesfully segmented lesion area from backgrounds in general and achieves a ideal result in the later three example. 

![](./images/pred_1.png)

![](./images/pred_2.png)

Figure 7, Four model predicted examples

## Dependencies
- Python 3.7
- TensorFlow 2.6.0
- scikit-learn 1.0.1
- matplotlib 3.4.3
- ISICs 2018 Challenge dataset (accessible at the following [link](https://challenge2018.isic-archive.com/)).


## References

[1] 2021. [Online]. Available: https://challenge2018.isic-archive.com/. [Accessed: 31- Oct- 2021].

[2] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Retrieved from: https://arxiv.org/pdf/1505.04597.pdf.


