# Segment the ISICs data set with the Improved UNet

## The Dataset
Our dataset is ISIC 2018 challenge data for skin cancer - This is part of the ISIC 2018 challenge and comes with segmentation labels. 
The following images is the example of the original image and the corresponding ground truth image.

![image](https://github.com/TTTZP/PatternFlow/blob/topic-recognition/recognition/S4549518/Images/Image_1.png)

## Model

### Original U-net model
U-Net is one of the earliest algorithms for semantic segmentation using full convolutional networks. The symmetric U-shaped structure that contains compression paths and expansion paths in the paper was very innovative at the time, and to some extent affected the following several The design of the segmentation network, the name of the network is also taken from its U-shaped shape.

The U-shaped structure of U-Net is shown in the figure below. The network is a classic fully convolutional network (that is, there is no fully connected operation in the network). The input of the network is a 572x572 image (input image tile) whose edges are mirrored. The left side of the network is a series of downsampling operations composed of convolution and Max Pooling. This part is called contracting in the paper path. The contracting path consists of 4 blocks, each block uses 3 effective convolutions and 1 Max Pooling downsampling. After each downsampling, the number of Feature Maps is multiplied by 2, so there is the Feature Map size shown in the figure. Variety. Finally, a feature map with a size of 32x32 is obtained.

The right part of the network is called the expansive path in the paper. It is also composed of 4 blocks. Before the start of each block, the size of the Feature Map is multiplied by 2 through deconvolution, and the number is halved (the last layer is slightly different), and then the Feature of the symmetrical compression path on the left Map is merged. Because the size of the Feature Map of the compressed path on the left and the expanded path on the right are different, U-Net normalizes the Feature Map by clipping the Feature Map of the compressed path to the Feature Map of the same size as the extended path (ie 1). The convolution operation of the extended path still uses the effective convolution operation, and the final Feature Map size is 388x388. Since the task is a two-classification task, the network has two output Feature Maps.

![image](https://github.com/TTTZP/PatternFlow/blob/topic-recognition/recognition/S4549518/Images/Image_2.png)

### Improved U-net model
From the essay "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge", we get a improved U-NET model. There are several adjustments compared to the original U-net model. A network architecture of the Improved U-Net is shown below.

![image](https://github.com/TTTZP/PatternFlow/blob/topic-recognition/recognition/S4549518/Images/Image_3.png)

There are 4 important adjustments:
 * Use context modules to compute he activations in the context pathway. Each context module is in fact a pre-activation residual block with two 3x3x3 convolutional layers and a dropout layer in between.
 * Use convolutional layers with stride 2 do downsampling, instead of using max pooling operations in original U-net model.
 * Following the concatenation, a localization module recombines these features together. A localization moduleconsists of a 3x3x3 convolution followed by a 1x1x1 convolution that halves the number of feature maps.
 * There are three additional segmentation layers play importtant role in the expansive path.
 
# Dependencies

### Usage of the ImprovedUnet_Demo.py file
In order run the demo for a deep understanding of whole project, we must download the dataset which is given by our tutor. The download link is "https://cloudstor.aarnet.edu.au/sender/?s=download&token=f0d763f9-d847-4150-847c-e0ec92d38cc5". There is a 3GB zip file called "ISIC2018_Task1-2_Training_Data.zip". After decompressing this zip file, there will be two files called "ISIC2018_Task1-2_Training_Input_x2" and "ISIC2018_Task1_Training_GroundTruth_x2". Also, this file will show the log of whole project. Besides, this file also can be used as drive script.

### Usage of use improved_Unet_model.py file
It will return a improved U-net model. which is a tensorflow object. This model is realized by referring to the above ideas. The input size is (256,256,3).

### Usage of evaluate_model.py file
This file is to evaluate the performance of model. There are several steps:
* Import data and resize image.
* Split data into training dataset and test dataset.
* Write the function of Dice similarity coefficient for evaluate model.
* Train the model on both training dataset and test dataset.
* Plot and compare the origin image, ground truth and the predict image.

What needs to be pointed out is the training dataset and test dataset. The original image is taken from "ISIC2018_Task1-2_Training_Input_x2" file and the ground truth is taken from "ISIC2018_Task1_Training_GroundTruth_x2". After import this image, I split them into training dataset and test dataset, the ratio of them is 0.8. Both training dataset and test dataset contain two sub datasets, one is the training inputs, and the other one is the ground truth. The size of training dataset and training dataset ground truth is (2075, 256, 256,3) and (2075, 256, 256,1), the size of test dataset and test dataset ground truth is (519, 256, 256,3) and (519, 256, 256,1).

# Result
After using improved model, we get the result of segemantation on test dataset. One example of result as below:

![image](https://github.com/TTTZP/PatternFlow/blob/topic-recognition/recognition/S4549518/Images/Image_4.png)


# References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1
