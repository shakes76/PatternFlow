# Improved UNet for ISICs dataset - Task 1

#####################################
This project is created for COMP3710 Pattern Recognition Report. 

This project aim to segment the ISICs data set with the Improved UNet model with all labels having a minimum Dice Similarity Coefficient (DSC) of 0.8 on the test set - Task 1 

This project is finished by Xinwei Li, 44935209.

#####################################
## Introduction

This program aims to solve a segmentation problem: classify pixels in an image by groups. Thus, the problem is to indicate lesion vs healthy skin, and there are only 2 groups, which is also known as the binary segmentation problem. A convolutional neural network outputs an original image as a mask, segmented into the yellow and dark purple 2D image. Specifically, the parameters of the U-Net model were tuned for the ISIC 2018: Skin Lesion Analysis- this U-Net model is competent in segmenting skin lesions with an average Dice Similarity Coefficient (DSC) of around 0.91 (result could double check-in demo.ipynb)

## Dataset

Assuming users who run this model have downloaded and unzipped the [ISICs 2018 dataset](https://cloudstor.aarnet.edu.au/sender/?s=download&token=723595dd-15b0-4d1e-87b8-237a7fe282ff).  This data set contains 2595 images of skin lesions and their respective segmentation labels, identifying the presence of lesions in the image. The segmentation labels are binary mask images in png format, as mentioned before,  identifying the location of the primary skin lesion within input lesion images, which contain either a pixel value of 0 for areas of the skin existing outside a lesion or 255 for areas inside a lesion.

There is an example of an image pair shown below. 

![training_input](https://github.com/Liz-Xinwei/PatternFlow/blob/topic-recognition/recognition/s44935209/ISIC_images/ISIC_0000001.jpg)

![segmentation mask](https://github.com/Liz-Xinwei/PatternFlow/blob/topic-recognition/recognition/s44935209/ISIC_images/ISIC_0000001_segmentation.png)

## Methodology

### UNet 

U-Net is one of the famous image segmentation structures used mainly for biomedical purposes. The name U-Net is because its structure contains a compressive path and an expansive path which can be viewed as a U shape. This architecture is built to generate better results even for a small group of training data set. There are several U-Net structures, from basic [U-Net](https://arxiv.org/pdf/1505.04597.pdf) to [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf),  [Attention U-Net](https://arxiv.org/pdf/1804.03999.pdf), [Attention R2U-Net](https://github.com/LeeJunHyun/Image_Segmentation), and UNet++, also called [Nested U-Net](https://arxiv.org/pdf/1807.10165.pdf). 

Image shown below is the algorithm structure of **U-Net**.

![UNet](https://github.com/Liz-Xinwei/PatternFlow/blob/topic-recognition/recognition/s44935209/result_images/unet1.png)

In my algorithm, this U-Net model mainly contain two block, convolution block and up convolution block.
#### Convolution Block
- Conv2d
- BatchNorm2d
- Relu
- Conv2d
- BatchNorm2d
- Relu

#### Up Convolution Block
- Upsample
- Conv2d
- BatchNorm2d
- ReLU

#### U-Net Model
- Convolution Block
- Maxpool
- Convolution Block
- Maxpool
- Convolution Block
- Maxpool
- Convolution Block
- Maxpool
- Convolution Block
- Up Convolution Block
- Convolution Block
- Up Convolution Block
- Convolution Block
- Up Convolution Block
- Convolution Block
- Up Convolution Block
- Conv2d 1 x 1
 

### Algorithm Models

This is the print view of my U-Net model. You could double check is in demo.ipynb or in models.py for more details.

To confirm the accuracy of this model, the Dice coefficient and IoU are calculated for all test predictions. Since, task 1 only asked to show the Dice coefficient, in README.md I will only discuss the DSC.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 128, 128]           1,792
       BatchNorm2d-2         [-1, 64, 128, 128]             128
              ReLU-3         [-1, 64, 128, 128]               0
            Conv2d-4         [-1, 64, 128, 128]          36,928
       BatchNorm2d-5         [-1, 64, 128, 128]             128
              ReLU-6         [-1, 64, 128, 128]               0
        conv_block-7         [-1, 64, 128, 128]               0
         MaxPool2d-8           [-1, 64, 64, 64]               0
            Conv2d-9          [-1, 128, 64, 64]          73,856
      BatchNorm2d-10          [-1, 128, 64, 64]             256
             ReLU-11          [-1, 128, 64, 64]               0
           Conv2d-12          [-1, 128, 64, 64]         147,584
      BatchNorm2d-13          [-1, 128, 64, 64]             256
             ReLU-14          [-1, 128, 64, 64]               0
       conv_block-15          [-1, 128, 64, 64]               0
        MaxPool2d-16          [-1, 128, 32, 32]               0
           Conv2d-17          [-1, 256, 32, 32]         295,168
      BatchNorm2d-18          [-1, 256, 32, 32]             512
             ReLU-19          [-1, 256, 32, 32]               0
           Conv2d-20          [-1, 256, 32, 32]         590,080
      BatchNorm2d-21          [-1, 256, 32, 32]             512
             ReLU-22          [-1, 256, 32, 32]               0
       conv_block-23          [-1, 256, 32, 32]               0
        MaxPool2d-24          [-1, 256, 16, 16]               0
           Conv2d-25          [-1, 512, 16, 16]       1,180,160
      BatchNorm2d-26          [-1, 512, 16, 16]           1,024
             ReLU-27          [-1, 512, 16, 16]               0
           Conv2d-28          [-1, 512, 16, 16]       2,359,808
      BatchNorm2d-29          [-1, 512, 16, 16]           1,024
             ReLU-30          [-1, 512, 16, 16]               0
       conv_block-31          [-1, 512, 16, 16]               0
        MaxPool2d-32            [-1, 512, 8, 8]               0
           Conv2d-33           [-1, 1024, 8, 8]       4,719,616
      BatchNorm2d-34           [-1, 1024, 8, 8]           2,048
             ReLU-35           [-1, 1024, 8, 8]               0
           Conv2d-36           [-1, 1024, 8, 8]       9,438,208
      BatchNorm2d-37           [-1, 1024, 8, 8]           2,048
             ReLU-38           [-1, 1024, 8, 8]               0
       conv_block-39           [-1, 1024, 8, 8]               0
         Upsample-40         [-1, 1024, 16, 16]               0
           Conv2d-41          [-1, 512, 16, 16]       4,719,104
      BatchNorm2d-42          [-1, 512, 16, 16]           1,024
             ReLU-43          [-1, 512, 16, 16]               0
          up_conv-44          [-1, 512, 16, 16]               0
           Conv2d-45          [-1, 512, 16, 16]       4,719,104
      BatchNorm2d-46          [-1, 512, 16, 16]           1,024
             ReLU-47          [-1, 512, 16, 16]               0
           Conv2d-48          [-1, 512, 16, 16]       2,359,808
      BatchNorm2d-49          [-1, 512, 16, 16]           1,024
             ReLU-50          [-1, 512, 16, 16]               0
       conv_block-51          [-1, 512, 16, 16]               0
         Upsample-52          [-1, 512, 32, 32]               0
           Conv2d-53          [-1, 256, 32, 32]       1,179,904
      BatchNorm2d-54          [-1, 256, 32, 32]             512
             ReLU-55          [-1, 256, 32, 32]               0
          up_conv-56          [-1, 256, 32, 32]               0
           Conv2d-57          [-1, 256, 32, 32]       1,179,904
      BatchNorm2d-58          [-1, 256, 32, 32]             512
             ReLU-59          [-1, 256, 32, 32]               0
           Conv2d-60          [-1, 256, 32, 32]         590,080
      BatchNorm2d-61          [-1, 256, 32, 32]             512
             ReLU-62          [-1, 256, 32, 32]               0
       conv_block-63          [-1, 256, 32, 32]               0
         Upsample-64          [-1, 256, 64, 64]               0
           Conv2d-65          [-1, 128, 64, 64]         295,040
      BatchNorm2d-66          [-1, 128, 64, 64]             256
             ReLU-67          [-1, 128, 64, 64]               0
          up_conv-68          [-1, 128, 64, 64]               0
           Conv2d-69          [-1, 128, 64, 64]         295,040
      BatchNorm2d-70          [-1, 128, 64, 64]             256
             ReLU-71          [-1, 128, 64, 64]               0
           Conv2d-72          [-1, 128, 64, 64]         147,584
      BatchNorm2d-73          [-1, 128, 64, 64]             256
             ReLU-74          [-1, 128, 64, 64]               0
       conv_block-75          [-1, 128, 64, 64]               0
         Upsample-76        [-1, 128, 128, 128]               0
           Conv2d-77         [-1, 64, 128, 128]          73,792
      BatchNorm2d-78         [-1, 64, 128, 128]             128
             ReLU-79         [-1, 64, 128, 128]               0
          up_conv-80         [-1, 64, 128, 128]               0
           Conv2d-81         [-1, 64, 128, 128]          73,792
      BatchNorm2d-82         [-1, 64, 128, 128]             128
             ReLU-83         [-1, 64, 128, 128]               0
           Conv2d-84         [-1, 64, 128, 128]          36,928
      BatchNorm2d-85         [-1, 64, 128, 128]             128
             ReLU-86         [-1, 64, 128, 128]               0
       conv_block-87         [-1, 64, 128, 128]               0
           Conv2d-88          [-1, 1, 128, 128]              65
================================================================
Total params: 34,527,041
Trainable params: 34,527,041
Non-trainable params: 0
```

### Project Structure

In this repository, there are serveral Python scripts used for different conditions:

- train.py - Trainning script that model is trained based on a pretrained 25 epoches one. If you want to check the model, please set a higher number of epoch! 
- test.py - The main driver script when running the model. 
- Data_Loader.py - Image data preprocessing. Getting individual transformations and data (as a Dict)
- image.py - Imgae data preprocessing, creating 2D image from 3D.
- Metrics.py - Calculates the dice coefficient for the images and getting the accuracy of the model.
- losses.py - Calculating the dice loss and metrics
- Models.py - The implemenation of the improved U-Net model
- ploting.py - Generating the output images and metric plot.
- Unet_ISICs_JupyterBook.ipynb - Additional Jupyter Notebook for displaying the esssential process step by step.

### Dependencies

1. python>=3.6
2. torch>=0.4.0
3. torchvision
4. torchsummary
5. tensorboardx
6. natsort
7. numpy
8. pillow
9. scipy
10. scikit-image
11. sklearn

## Result

### Parameters and Training, Validation, Testing

The input images, which initially had different sizes, were **resized** to 128 x 128 for faster training. The training data was split between training: validation: testing on 80:15:5 ratio. **Batch size** of 4 and **Adam optimizer learning rate** of 0.001 was utilized., which gives satisfactory results; a **validation DSC** of 0.9147 was obtained after 6 epochs and, consequently, the trained model gives an **average DSC** of 0.9122 on the test dataset.

### Result images and Plots

At the end of this algorithm, the average DSC are around 0.91. The result of test image could also demonstrate this fact.

1. Below image: previous image -> resize -> generated -> target -> real target

![below image: previous image -> resize -> generated -> target -> real target](https://github.com/Liz-Xinwei/PatternFlow/blob/topic-recognition/recognition/s44935209/result_images/test%20image.png)

2. Learning curve: Train loss -> Valid loss. These two plots for the first 25 epochs initially the train loss and valid loss are droped rapidly from 0.40 to 0.2 (around), then it slowly but steadly go all the way down to 0.10 in train loss and valid loss fluctuates between 0.22 and 0.175.
 
![Learning curve: Train loss -> Valid loss](https://github.com/Liz-Xinwei/PatternFlow/blob/topic-recognition/recognition/s44935209/result_images/loss_image.png)

3. Here, indicate the dice similarity coefficient and the intersection over union result on the test set. DSC owned a stable value range from 0.90 to 0.95, which started from 7 epochs. Same as DSC, IoU also holds a constant range, 0.8 - 0.9, from 7th epochs.

![DSC -> IoU](https://github.com/Liz-Xinwei/PatternFlow/blob/topic-recognition/recognition/s44935209/result_images/Tets_Dice_IoU.png)

### Reference
1. O.Ronneberger, P.Fischer, and T.Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," May 2015. [Online]. Available:https://arxiv.org/abs/1505.04597
2. M.Z.Alom, M. Hasan, C. Yakopcic, T. M. Taha, V. K. Asari, "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation," Feb. 2018. [Online] Available: https://arxiv.org/abs/1802.06955
 
