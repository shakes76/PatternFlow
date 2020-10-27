# Segment the ISICs data set with the Improved UNet
Image segmentation of ISICs data set implemented for tensorflow

## Description

ISICs data set concludes thousands of Skin Lesion images. This recognition algorithm aims to automatically do Lesion Segmentation through an improved unet model [[1]](#References). The evaluation of segmentation has a [Dice similarity coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient) of 0.84 on the test set. 

Here is a exmaple of the original Skin Lesion image and its Ground Truth segmentation image.

​                                 <img src="images/ISIC_0000000.jpg" alt="ISIC_0000000" style="zoom:50%;" />                                    <img src="images/ISIC_0000000_segmentation.png" alt="ISIC_0000000_segmentation" style="zoom:50%;" />  

<center>
  Figure 1. Left is original Skin Lesion image and the right is its ground truth image.
</center>

To conclusion, the aim of this project is to create the segmentation image like the above right one from an input image like the above left one by a kine of uNet model.

## uNet Structure

The uNet Structure is almostly from the structure in [[1]](#References).

![uNetStructure](images/uNetStructure.jpg)

<center>
  Figure 2. The structure of the improved uNet 
</center>

The difference is that images in ISICs data set are 2D dimensions, and so all $3\times 3\times 3$ convolution layers have been changed to $3\times 3$ convolution layers.

**Context module**: InstanceNormalization, ReLu, $3\times 3$ convolution, InstanceNormalization, ReLu, $3\times 3$ convolution, Dropout(0.3).

​								 It is implemented as a method called “context_modules” in model.py.

**Upsampling module**: UpSampling2D(), $3\times 3$ convolution

​										 It is implemented as a method called “upSampling” in model.py.

**Localization model**: $3\times 3$ convolution, $1\times 1$ convolution

​									   It is implemented as a method called “localization” in model.py.

**Segmentation layer**: $1\times 1$ convolution by 2 output filters

**Upscale**: UpSampling2D() by bilinear interpolation

**Optimizer**: Adam

**Loss**: sparse_categorical_crossentropy

**Metrics**: accuracy

The whole model is implemented as a method called “improvedUnet” in model.py.

## Workflow

![WorkFlow](images/WorkFlow.jpg)

##  Dependencies required

## Example outputs and plots

![Example_Output](images/Example_Output.png)

In most cases, the pridiction is almose the same as the ground truth segmentation image and can segment Skin Lesion correctly.

## Interesting finding

![interests](images/interests.png)

The above plots show that sometimes the prediction image is better than the ground truth image. The ground truth image may be wrong due to human negligence.

## Splitting data set

I split data set into training set, validation set and testing set.

Firstly, I split the whole dataset into (training,  validation) set and testing set, which the proportions are 0.8 and 0.2 respectively.

Then I split the (training,  validation) set into training set and validation set, which the proportions are 0.8 and 0.2 respectively.

Hence, the total proportions of training set, validation set and testing set are 0.64, 0.16, 0.2.

## References

[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1

