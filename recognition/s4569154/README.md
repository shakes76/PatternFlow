README (Runqi Lin, 45691548)
====
Background
-------
Detect lesions within the ISICs data set with a YOLO network such as the original YOLO with all detections having a minimum Intersection Over Union of 0.8 on the test set and a suitable accuracy for classification.

ISIC 2018 challenge data for skin cancer - This is part of the ISIC 2018 challenge and comes with segmentation labels. The preprocessed version of this data set can be found on the course Blackboard site.

Datasets
-------
ISIC2018_Task1-2_Training_Data
ISIC2018_Task1-2_Test_Input

To test the model I trained, I also downloaded the corresponding test dataset containing 1000 images.
https://challenge.isic-archive.com/data/

Yolov3:Real-Time Object Detection
-------
YOLOv3 is extremely fast and accurate. In mAP measured at .5 IOU YOLOv3 is on par with Focal Loss but about 4x faster. Moreover, you can easily tradeoff between speed and accuracy simply by changing the size of the model, no retraining required!

The main improvements of YOLO3 are: adjusting the network structure; using multi-scale features for object detection; and replacing softmax with logistic for object classification.

darknet-53 adopts the idea of resnet and adds a residual module to the original network, which is helpful to solve the gradient problem of the deep network, and each residual module consists of two convolutional layers and a shortcut connections,
Compared with yolov1 and v2, v3 has no pooling layer and no full connection layer. The down-sampling of the network is achieved by setting the stride of the convolution to 2, and the size of the image is reduced to half after each convolutional layer. The implementation of each convolutional layer consists of convolution + BN + Leaky relu , and each residual module is followed by a zero padding, as shown in the figure below.

![image](https://user-images.githubusercontent.com/75237235/139426179-403ad5cd-491b-42e4-9ab1-731950adb6df.png)



