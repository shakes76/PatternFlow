# Image Segmentation Task
Task: Segment the OASIS brain data set with an Improved UNet with all labels having a minimum Dice
similarity coefficient of 0.9 on the test set.

[Data resource](https://cloudstor.aarnet.edu.au/plus/s/n5aZ4XX1WBKp6HZ): magnetic resonance (MR) image of the brain from OASIS dataset.

# Load Dataset
x_train dataset has 9664 images which are 256*256 pixels(black and white photo)

x_test dataset has 544 images which are 256*256 pixels(black and white photo)

y_train (labels) has 9664 images which are 256*256 pixels(black and white photo)

y_test (labels) has 544 images which are 256*256 pixels(black and white photo)

![](images/example.png)

# Data preparation

 Using **sorted function** to sort each dataset to make sure that each picture can correspond to the corresponding label. After that, using **reshape function** to make x_train data and x_test data into 4 dimensions for instance (9664,256,256,1) and (544,256,256,1), and then divide training data and test data by 255 to achieve **normalization**. In y_train data and y_test data has 4 unique number which are 0, 85, 170 and 255, and dividing y_train data and y_test_data by 85 to make 4 labels which are 1, 2, 3 and 4.


### 4 labels which are:
* 0 - Background
* 1 - CSF (cerebrospinal fluid)
* 2 - Gray matter
* 3 - White matter

# For example:
![](images/labels.png)

# Unet Model
![](images/UNET.jpg)
The shape of Unet network structure is similar to U like the picture above. It contains convolution and pooling layer. The left half is the encoder which is down-sampling in the traditional classification network, and the right half is the decoder is the up-sampling. The gray arrow in the middle is a jump connection, which captures the shallow features with the deep features, because the shallow layer can usually capture some simple features of the image, such as borders and colors. The deep convolution operation captures some unexplainable abstract features of the image. It is best to use the shallow and deep at the same time, while also allowing the decoder to learn to lose relevant features in the encoder pooling downsampling.

# UNET++
![](images/unet++.png)

