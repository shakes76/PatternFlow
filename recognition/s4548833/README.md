# Implementation of the YOLO of ISIC dataset

## Problem:
This project is to detect lesions within the ISICs data set by YOLO network.

## List ofDependencies
darknet
opencv2
os

## Model Architecture
The main idea of YOLO is regressing the position of the bounding box and the category to which the bounding box belongs directly in the output layer (the whole picture is used as an input of the model and It convert the Object Detection problem into a Regression problem). It is composed of 3 steps: Zoom image, pass the image through a fully convolutional neural network and flitering with maximum suppression (NMS).

In this project, I use the YOLOV3 with the darknet53. This network is mainly composed of a series of 1x1 and 3x3 convolutional layers (each convolutional layer will be followed by a Batch normalization layer and a LeakyReLU) layer. The author said that because there are 53 convolutional layers in the network, it is called Darknet- 53 (2 + 1*2 + 1 + 2*2 + 1 + 8*2 + 1 + 8*2 + 1 + 4*2 + 1 = 53 According to the sequence number, the last Connected is a fully connected layer, which is also considered a convolution There are 53 layers in total). The figure below is the structure of Darknet-53
![image](https://user-images.githubusercontent.com/93363361/142148254-f8833af9-95f8-428e-99a4-ed9128673590.png)

The YOLOv3 network uses (4+1+c) k convolution kernels of size 11 to perform convolution prediction in the three feature maps, and k is the number of preset bounding box priors (k defaults to 3 ), c is the number of categories of the predicted target, 4k parameters are responsible for predicting the offset of the target bounding box, k parameters are responsible for predicting the probability of the target contained in the bounding box of the target, and ck parameters are responsible for predicting the k preset bounding boxes Corresponding to the probability of c target categories. The following figure shows the prediction process of the target bounding box. The dotted rectangular box in the figure is the preset bounding box, and the solid rectangular box is the predicted bounding box obtained by calculating the offset predicted by the network. The conversion process from the preset bounding box to the final predicted bounding box is shown in the formula on the right, where the function is the sigmoid function whose purpose is to scale the predicted offset to between 0 and 1.
![image](https://user-images.githubusercontent.com/93363361/142149089-df9dd514-9323-46d5-a7ce-89c42f45049b.png)

The main idea of YOLO is regressing the position of the bounding box and the category to which the bounding box belongs directly in the output layer (the whole picture is used as an input of the model and It convert the Object Detection problem into a Regression problem). It is composed of 3 steps: Zoom image, pass the image through a fully convolutional neural network and flitering with maximum suppression (NMS).


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
                                                                                                                                          
