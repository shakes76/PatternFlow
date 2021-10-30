Implementation of the YOLO of ISIC dataset
====
Problem:
====
This project is to detect lesions within the ISICs data set by YOLO network.
Coding environment and requirements:
====
google colab python 3.8

Yolov3 by darknet package

ModelArchitecture
====
![image](https://user-images.githubusercontent.com/93363361/139520081-f36e8769-8bc7-4ff9-b7b0-62c57c17fc1c.png)

![image](https://user-images.githubusercontent.com/93363361/139520451-9d02fa5f-4928-4e18-b63d-f2738a47a0bf.png)

![image](https://user-images.githubusercontent.com/93363361/139520459-27524030-6f50-430a-83bc-bbc698c41388.png)

The main idea of YOLO is regressing the position of the bounding box and the category to which the bounding box belongs directly in the output layer (the whole picture is used as the input of the network to convert the Object Detection problem into a Regression problem).
