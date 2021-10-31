
# Detect lesions within the ISICs data set with the Yolov3 network


## Algorithm Description


Yolo v3 extracts features from the input features through a feature extraction network, and obtains a feature map output of a specific size. The input image is divided into 13×13 grid cells, and then if the center coordinates of an object in the real frame fall in a certain grid cell, then the grid cell will predict the object. Each object has a fixed number of bounding boxes. There are three bounding boxes in YOLO v3. Logistic regression is used to determine the regression box used for prediction.
![image](https://user-images.githubusercontent.com/69140217/139566820-d3d4a1dc-9872-477f-a44f-95b1253d7b7d.png)

## Dependencies required
1.CUDA:Development environment for GPU computing
2.OpenCV: show image
3.OS
4.darknet framework
Darknet is an open source neural network framework written in C and CUDA.

## Example outputs
We can see that the predict result of one of test images

![image](https://user-images.githubusercontent.com/69140217/139567868-e4339825-834f-4d1d-8750-6a923e183885.png)

For the final results.You can find it at results.txt.

![1635655274(1)](https://user-images.githubusercontent.com/69140217/139568055-7659986c-575b-4b33-8712-c5809fb82342.jpg)

## Train and Test data
Using 2018 ISIC  data from bb as the train data.It has 2594 images
Using 2016 ISIC test data as the test data.It has 379 images.

## Implementation process

First, I clone darknet from AlexeyAB's famous repository, adjust the Makefile to enable OPENCV and GPU for darknet and then build darknet.
Then we need to connect to google drive to better upload the various files we need to run the code.
After configuring the operating environment, we first need to label the dataset. In order to make the labeling more accurate, we input the GroundTruth image and use the ISIC_Label.ipynb file for feature extraction. Each image (whether it is train data or test data) is generated accordingly There are five parameters in a text file. The first parameter is class. In this project, only Lesions are considered, so classes=0. The second parameter represents the value of the x center of the extraction box, and the third parameter is the value of the y_center of the extraction box. The fourth parameter is the width of the extraction frame. The fifth parameter is the height of the extraction frame.
![image](https://user-images.githubusercontent.com/69140217/139571390-2f0a59f3-6b16-4b98-95fb-ed155997d04d.png)
![image](https://user-images.githubusercontent.com/69140217/139571400-d11a69e3-bb13-4c01-ae46-a99b49ef85b0.png)


Put the obtained txt file distribution into train data and test data. After compression, upload it to Google drive.
In addition, we need to create two files: obj.data and obj.name. The former is used to store the paths of the training set and the test set. The latter represents the name of the recognition class, here only Lesions.
Moreover, we also need a cfg file. for training and testing the model. We can download it from the darknet framework and modify it.
After many adjustments, the yolov3_custom3.cfg file is obtained.
![image](https://user-images.githubusercontent.com/69140217/139571487-5c9b6bd5-bf74-4235-8fd8-55ef9dfc52ae.png)


The last configuration file we need before we start training our detector is the train.txt file, which contains the relative paths of all our training images.
Use generate_train.py and generate_test.py files to generate train.txt and test.txt. Convenient for follow-up training and testing.
Then we can train the model.This process needs about 10 hours.(Depends on max_batch)

## Reference
[1]https://www.youtube.com/watch?v=10joRJt39Ns
[2] https://blog.csdn.net/mieleizhi0522/article/details/79919875
[3]https://pjreddie.com/darknet/
[4]https://zhuanlan.zhihu.com/p/91587361
