# Segement the ISICs data set with the Improved UNet

This project is aimed to train a improved UNet model for detecting the melanoma shape from its photo. After properly trained, the model achieves the accuracy over 0.92 and dice similarity coefficient (DSC) over 0.8.

## BACKGROUND
The International Skin Imaging Collaboration is an internationa effrort to improve melanoma disgnosis. The project target is to develop image analysis tools to enable the auto mate disgnosis of melanoma from dermoscopic images [1]. The challenge is divided into three sub tasks which are:
Task 1: Lesion Segmentation 
Task 2: Lesion Attribute Detection 
Task 3: Disease Classification 

In our project, only task one is focused and addressed. Our model use Imporved UNet to segement the lesion area from background with a high accuracy and DSC. 

## DATA SET DESCRIPTION

The datasets used in this project from training, validating and testing are all from ISICs 2018 Challenge Data set. The datasets contains two matched groups with each group has 2594 images. The first group of images is the raw melanoma images and the second groups of image (mask image) is the segmentation of the lesion area which will serve as labels in following model training.  mask images are binary images with only pixel value of 0 and 255. Pixel value = 255 means the lesion area while 0 means background area. Figure 1 is an example of melanoma raw image (group 1 image) and Figure 2 is the responding mask image.

![](./images/ISIC_0000008.jpg)
*Figure 1, An example of melanoma raw image

![](./images/ISIC_0000008_segmentation.png)
*Figure 2, Mask image (segmentation image) of Figure 1 


## DATA PREPROCESSING

Data preprocessing of this project including four steps:

1. load raw data

In this project, the grayscale version of images are loaded for training in the model 

2. split raw data into training set, validating set and test set 

3. decode the image into predefined size 256*192

4. map the training image and segmentation masks

## Model Description

### Model Structure
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 256, 192, 1) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 256, 192, 16) 160         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 256, 192, 16) 2320        conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 128, 96, 16)  0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 96, 32)  4640        max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 128, 96, 32)  9248        conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 64, 48, 32)   0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
dropout (Dropout)               (None, 64, 48, 32)   0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 48, 64)   18496       dropout[0][0]                    
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 48, 64)   36928       conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 32, 24, 64)   0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 24, 128)  73856       max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 24, 128)  147584      conv2d_6[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 32, 24, 128)  0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 16, 12, 128)  0           dropout_1[0][0]                  
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 12, 256)  295168      max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 12, 1024) 2360320     conv2d_8[0][0]                   
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 32, 24, 1024) 0           conv2d_9[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 24, 1152) 0           conv2d_7[0][0]                   
                                                                 up_sampling2d[0][0]              
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 24, 128)  1327232     concatenate[0][0]                
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 24, 128)  147584      conv2d_11[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 48, 128)  0           conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 64, 48, 64)   73792       up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 64, 48, 128)  0           conv2d_5[0][0]                   
                                                                 conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 64, 48, 64)   73792       concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 64, 48, 64)   36928       conv2d_14[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 128, 96, 64)  0           conv2d_15[0][0]                  
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 128, 96, 32)  18464       up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 96, 64)  0           conv2d_3[0][0]                   
                                                                 conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 128, 96, 32)  18464       concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 128, 96, 32)  9248        conv2d_17[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 256, 192, 32) 0           conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 256, 192, 16) 4624        up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 192, 32) 0           conv2d_1[0][0]                   
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 256, 192, 16) 4624        concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 256, 192, 16) 2320        conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 256, 192, 1)  17          conv2d_21[0][0]                  
Total params: 4,665,809
Trainable params: 4,665,809
Non-trainable params: 0


## First Epochs Result
![图片](https://user-images.githubusercontent.com/31636541/139171551-fe47b000-0b8b-482d-ac9d-4008f77a6e32.png)
## 100TH Epochs Result
![图片](https://user-images.githubusercontent.com/31636541/139171627-fc64917d-0c09-4487-a663-f24ede88190f.png)
## 200TH Epochs Result
![图片](https://user-images.githubusercontent.com/31636541/139171664-5765ebd6-e344-44f6-b78e-f7dd622bcad3.png)


## Training History

### Training Accuracy vs Validation Accuracy
![图片](https://user-images.githubusercontent.com/31636541/139171770-2e25555b-d383-4772-9dd7-95bf6120ad89.png)
### Training Dice Coefficient vs Validation Dice Coefficient
![图片](https://user-images.githubusercontent.com/31636541/139171822-8339d4bb-7b98-42f8-948c-b3918ae2caea.png)


# RESULTS
### Four sets images with the first image being raw image, the second one being raw mask and the third one being predicted mask
![图片](https://user-images.githubusercontent.com/31636541/139170066-d00a2ad1-0918-4d08-b7dc-172b2d8e273c.png)
![图片](https://user-images.githubusercontent.com/31636541/139170087-35f4f46b-c6fb-4970-9e00-5c3c4d255408.png)
![图片](https://user-images.githubusercontent.com/31636541/139170100-fa83f098-edc4-4018-bc6b-a3bca900978d.png)
![图片](https://user-images.githubusercontent.com/31636541/139170110-f0a28387-325b-4d88-8aac-31ad1f579d9f.png)

