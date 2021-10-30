# Yolov3 ISIC dataset lesion detection

The skin is the largest organ of the human body. Some skin diseases, such as malignant black, melanoma. These diseases are likely to cause death, and the consequences of misdiagnosis will be very serious. Moles are skin diseases with mild symptoms, but some fatal skin diseases such as malignant black and mildly symptomatic skin diseases and their Similar. Effectively locating and detecting these skin diseases plays a critical role in saving the lives of patients. The ISIC data set contains 23k melanoma examples of malignant and benign images that classify skin injuries. Yolo is a kind of slider image The detection technology is different from mask-rcnn in that it uses a one-stage algorithm, which only uses a convolutional neural network to directly predict the classes and positions of different targets. In here, YoloV3 is used to detect the lesion area of the ISIC data set. The goal of this task is that all detections having a minimum Intersection Over Union of 0.8 on the test set.


* [ISIC dataset](#ISIC dataset)<br>
* [YoloV3](#YoloV3)<br>
* [Data pre-processing](#Data pre-processing)<br>
* [Results](#Results)<br>
* [Dependencies](#Dependencies)<br>
* [Reference](#Reference)<br>

### ISIC dataset
In this task, we have given the preprocessed data set, which includes 2594 pictures and the corresponding segmented black and white pictures. 
![ISIC_0016037](https://user-images.githubusercontent.com/65603393/138678240-f794ef8b-b534-4a91-a953-96f2cb366411.jpg)
*Figure 1: Training image.*

![ISIC_0016037_segmentation](https://user-images.githubusercontent.com/65603393/138678838-ce003a09-fb42-41b2-96dd-31819110ff42.png)
*Figure 1: Segmented image.*


### YoloV3
Yolov3 is based on the changes made in v2 and v1. The main improvements are: 1. Adjusted the network structure to use multiscale features for object detection 2. Object classification replaced softmax with logistic.
![image](https://user-images.githubusercontent.com/65603393/138707537-3a35da27-fb4a-49a5-ab36-e0b259372e13.png)
*Figure 3: Resnet skip connection.*

YoloV3 adopts Darknet53 for image feature extraction. In Darknet53, some convolutional layers adopt the practice of residual network, and some layers directly set up skip connections to make the network deeper and extract more features. The pooling layer is replaced by a convolution operation with a step size of 2 for deep extrating the features.
![image](https://user-images.githubusercontent.com/65603393/138707097-e2e51e03-f852-4be5-80e9-740be4a0f3a9.png)
*Figure 4: Darknet53 and upsampling.*

YoloV3 uses three different scales of feature maps for target detection to solve the problem that YoloV2 and V3 are insensitive to small targets, which are 1313, 26x26, and 52x52, respectively, to detect targets of three different sizes: large, medium, and small.
The feature fusion layer selects the three scales of feature maps produced by DarkNet as input, and fuses the feature maps of each scale through a series of convolution layers and upsampling.

YOLOv3 divides the input image into SxS lattices, and each lattice predicts B bounding boxes, each bounding box prediction includes: x, y, width, height, Confidence and the probability of C categories, so the number of channels in the output layer of YOLOv3 is Bx(5 + C), in here for the ISIC dataset that is 3x(5+1)=18, 3 means a grid cell contains 3 bounding boxes, 4 means the 4 coordinate information of the box, and 1 means Confidence score. 
### Data pre-processing
In the data pre-processing stage, the ISIC dataset segmentation image set is used. The first white pixel values (x_min,y_min) and (x_max,y_max) of each image were found during image pixel iterations and making to tags for the xml files. Also, when splitting the dataset, the training and test sets split into 80% and 20%. According to the YoloV3 data loading method, these training and test sets are made into absolute paths and stored in train.txt and test.txt for Dataloader.
Finally, in the process of loading the dataset in pytorch, when the imported data is the training set, the images will be enhanced to reduce the impact of insufficient data, improving the robustness of the model, providing various "invariants" to the model, and to increase the model's ability to resist overfitting, including resizing, adding gray bars, and image flipping. 
## Results
![loss3 14314](https://user-images.githubusercontent.com/65603393/139538340-83ee43cd-1e61-45f9-92a6-9bb96b90ca1b.png)

![loss4 73156](https://user-images.githubusercontent.com/65603393/139538254-ca583fa7-c163-4ff1-8e68-976c028614bb.png)
![loss2 26325](https://user-images.githubusercontent.com/65603393/139538438-f970684f-efc1-44ed-b161-b99ed08df998.png)




## Dependencies
* Python 3.7
* Numpy 1.20.3
* Torch 1.9.0
* Torchvision 0.4.0
* Opencv-Python 4.5.3.56
* PIL 1.1.7
* Matplotlib 3.1.2
## Reference
GitHub - bubbliiiing/yolo3-pytorch. (2020, September 9). GitHub. Retrieved October 15, 2020, from https://github.com/bubbliiiing/yolo3-pytorch
