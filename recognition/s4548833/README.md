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

The main idea of YOLO is regressing the position of the bounding box and the category to which the bounding box belongs directly in the output layer (the whole picture is used as an input of the model and It convert the Object Detection problem into a Regression problem). It is composed of 3 steps: Zoom image, pass the image through a fully convolutional neural network and flitering with maximum suppression (NMS).
For the first step, an input image is first divided into S × S grids of equal size, and each grid is called a grid cell. The second step is the bounding boxes predition. In this step, YOLO provides two prediction frames for each grid. These prediction frames are based on the center of the grid and have a custom size. Each grid predicts B bounding boxes which has 4 coordinateds and a confidence level. Therefore, the prediction is composed of S × S × (B ∗ 5 + C) vectors. The third step is to use the NMS to find the most appropriate frame.

## Dataset
Training set: ISIC 2018 training set, with 2594 images and the groudtruth greyscale imgs of each imgs.
Testing set: ISIC 2016 with 379 images the groudtruth greyscale imgs of each imgs.

## Processing
1. Connect to the darknet package
2. Adjust the configuration of the darknet, in the yolov3_custom.cfg file,  I change the batch to 128 and subdivision to 16 and random=1. For the obj.name, I add the label of the recognization, which is the only label called lesions. For the obj.data, I change the path of the function to load file.
3. Use the labelImg to get the range of the lesions of each image of the training set.
4. Train the model and the weights every 1000 batches.
5. Use the testing set to get the prediciton and the performance of the model.

## Outplots and Results
![image](https://user-images.githubusercontent.com/93363361/139567948-43addbfa-073f-45dc-8263-3e71c4ea4433.png)
![image](https://user-images.githubusercontent.com/93363361/139567899-aff38e3e-3b66-4b15-8d30-880895159926.png)

![image](https://user-images.githubusercontent.com/93363361/139567419-4dd0adb2-51f5-4acb-9a81-00d0742f63ef.png)
![image](https://user-images.githubusercontent.com/93363361/139567855-dcb00662-d9bd-455a-9bb1-c08978f5a405.png)

![image](https://user-images.githubusercontent.com/93363361/139567423-cae5341e-cf0c-417e-863f-9543981e16b1.png)
![image](https://user-images.githubusercontent.com/93363361/139567821-4d054d2e-9a81-4ae0-b97a-1b0c08c2e7af.png)

From the images, it is indicate that the shape of the frame of yolo is similar to the groundtruth greysclae images. In this project, I get the weights of the 3000 batches and the 4000 batches, but the result is far more different, for the 4000 batches, the IoU and F1-score are both 0, I consider ther reason is overfitting, when the model has 4000 batches, the number of weight learning iterations is enough (Overtraining) to fit the noise in the training data and the non-representative features in the training examples. And from the result_3000.txt file, the IoU threshold is 80%, which means the framed box does have the 80% confidence of the object and the framed box includes all the features of the entire object. The f1-score of the testing set is 0.86 and IoU is 75.94%, which means the performance of the model is good.
                                                                                                                                          
