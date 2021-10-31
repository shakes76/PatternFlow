# Implementation of the YOLO of ISIC dataset

## Problem:
This project is to detect lesions within the ISICs data set by YOLO network.

## List ofDependencies
darknet
opencv2
os


## Model Architecture

![image](https://user-images.githubusercontent.com/93363361/139520081-f36e8769-8bc7-4ff9-b7b0-62c57c17fc1c.png)

![image](https://user-images.githubusercontent.com/93363361/139520451-9d02fa5f-4928-4e18-b63d-f2738a47a0bf.png)

![image](https://user-images.githubusercontent.com/93363361/139520459-27524030-6f50-430a-83bc-bbc698c41388.png)

The main idea of YOLO is regressing the position of the bounding box and the category to which the bounding box belongs directly in the output layer (the whole picture is used as the input of the network to convert the Object Detection problem into a Regression problem). It is composed of 3 steps: Zoom image, pass the image through a fully convolutional neural network and flitering with maximum suppression (NMS).
For the first step, an input image is first divided into S × S grids of equal size, and each grid is called a grid cell. The second step is the bounding boxes predition. In this step, YOLO provides two prediction frames for each grid. These prediction frames are based on the center of the grid and have a custom size. Each grid predicts B bounding boxes, and each bounding box has four coordinates and a confidence level, so the final prediction result is S × S × (B ∗ 5 + C) vectors. The third step is to use the NMS to find the most appropriate frame.

## Dataset
Training set: ISIC 2018 training set, with 2594 images.
Testing set: ISIC 2016 with 379 images.

## Outplots and Results

![image](https://user-images.githubusercontent.com/93363361/139567416-64854552-0a20-4f83-87d6-44835b0ace45.png)
![image](https://user-images.githubusercontent.com/93363361/139567419-4dd0adb2-51f5-4acb-9a81-00d0742f63ef.png)
![image](https://user-images.githubusercontent.com/93363361/139567423-cae5341e-cf0c-417e-863f-9543981e16b1.png)


