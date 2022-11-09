Detect lesions within the ISICs data set with yolo-v3
==
## Problem:
This project mainly for lesion segmentation on ISICs data set 2018.

## Yolo-v3:
YOLO is the abbreviation of "You Only Look Once", it is a object detection model. Compared with other object detection model, it has speed advantage with acceptable accuracy. 
The main process of yolo-v3 is shown in followed picture:
![捕获](https://user-images.githubusercontent.com/93361859/139585661-f6a7aaa3-d912-46c1-9785-6178c7206413.PNG)  
Yolo-v3 uses darknet-53 as the feature extraction network, it contains 53 convolutional layers, each followed by a batch normalization layer and a leaky ReLU layer.
In the end the darknet-53 will generate 3 feature maps with different size, which can be used to predict targets of different size. 

## Main process:
1. Use label.py to generate center coordinate, the bounding boxes' width and length of train ground truth.
2. Use the train dataset and train label from step1 into yolo_train.ipynb, change the configuration file and then train the model.
3. Use label.py to generate center coordinate, the bounding boxes' width and length of test ground truth.
4. Use the trained yolo-v3 weights and test label from step3 into yolo_test.ipynb, compute the result

## Dependencies:
Python 3.7  
Darknet  
Os  
Opencv  
Matplotlib  

## Dataset:
Train dataset: ISIC2018_Task1-2_Training_Data, including 2594 images  
Test dataset: ISIC2016_Task1_Testing_Data, including 379 images

## Example output:
![1](https://user-images.githubusercontent.com/93361859/139583117-47baf06e-6c8c-4f8e-b51e-e4e36eec0e89.PNG)
![2](https://user-images.githubusercontent.com/93361859/139583128-a0307bb8-c6bc-47c4-b7cf-dd53f62f276e.PNG)
![3](https://user-images.githubusercontent.com/93361859/139583135-0b9c46f9-088f-49a0-bc48-043e95044e9c.PNG)

## Result:
After test, the yolov3_3.cfg achieved the best result, which is shown in followed picture:  
![9](https://user-images.githubusercontent.com/93361859/139585719-883289eb-e216-4aca-978d-181982eadd3d.PNG)  
Overall the result is good, the mean average precision = 0.8286.

## References:
[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-
Time Object Detection,” arXiv:1506.02640 [cs], May 2016, arXiv: 1506.02640. [Online]. Available:
http://arxiv.org/abs/1506.02640

[2] Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[3] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and ′ S. Belongie. Feature pyramid networks for object detection. In CVPR, 2017. 2, 4, 5, 7

[4] A.I.G.C. (2020). GitHub - theAIGuysCode/YOLOv3-Cloud-Tutorial: Everything you need in order to get YOLOv3 up and running in the cloud. Learn to train your custom YOLOv3 object detector in the cloud for free! GitHub. https://github.com/theAIGuysCode/YOLOv3-Cloud-Tutorial

[5] Redmon J, Farhadi A. YOLO9000: better, faster, stronger[J]. arXiv preprint, 2017.
