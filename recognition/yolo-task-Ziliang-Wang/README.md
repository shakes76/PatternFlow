# Yolov3 ISIC dataset lesion detection

The skin is the largest organ of the human body. Some skin diseases, such as malignant black, melanoma. These diseases are likely to cause death, and the consequences of misdiagnosis will be very serious. Moles are skin diseases with mild symptoms, but some fatal skin diseases such as malignant black and mildly symptomatic skin diseases and their Similar. Effectively locating and detecting these skin diseases plays a critical role in saving the lives of patients. The ISIC data set contains 23k melanoma examples of malignant and benign images that classify skin injuries. Yolo is a kind of slider image The detection technology is different from mask-rcnn in that it uses a one-stage algorithm, which only uses a convolutional neural network to directly predict the classes and positions of different targets. In here, YoloV3 is used to detect the lesion area of the ISIC data set. The goal of this task is that all detections having a minimum Intersection Over Union of 0.8 on the test set.

* [YoloV3](#YoloV3)<br>

* 
### ISIC dataset
In this task, we have given the preprocessed data set, which includes 2594 pictures and the corresponding segmented black and white pictures. 
![ISIC_0016037](https://user-images.githubusercontent.com/65603393/138678240-f794ef8b-b534-4a91-a953-96f2cb366411.jpg)
*Figure 1: Training image.*

![ISIC_0016037_segmentation](https://user-images.githubusercontent.com/65603393/138678838-ce003a09-fb42-41b2-96dd-31819110ff42.png)
*Figure 1: Segmented image.*


### YoloV3
Yolov3 is based on the changes made in v2 and v1. The main improvements are: 1. Adjusted the network structure to use multiscale features for object detection 2. Object classification replaced softmax with logistic.

YoloV3 adopts Darknet53 for image feature extraction. In Darknet53, some convolutional layers adopt the practice of residual network, and some layers directly set up skip connections to make the network deeper and extract more features. The pooling layer is replaced by a convolution operation with a step size of 2 for deep extrating the features.
![image](https://user-images.githubusercontent.com/65603393/138686146-0bb40969-816e-486f-96b3-798e7aff88e0.png)

YoloV3 uses three different scales of feature maps for target detection to solve the problem that YoloV2 and V3 are insensitive to small targets, which are 1313, 26x26, and 52x52, respectively, to detect targets of three different sizes: large, medium, and small.
The feature fusion layer selects the three scales of feature maps produced by DarkNet as input, and fuses the feature maps of each scale through a series of convolution layers and upsampling.

YOLOv3 divides the input image into SxS lattices, and each lattice predicts B bounding boxes, each bounding box prediction includes: x, y, width, height, Confidence and the probability of C categories, so the number of channels in the output layer of YOLOv3 is Bx(5 + C), in here for the ISIC dataset that is 3x(5+1)=18, 3 means a grid cell contains 3 bounding boxes, 4 means the 4 coordinate information of the box, and 1 means Confidence score. 

## Problem Description

