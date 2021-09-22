# Classify laterality of the OAI AKOA knee data set
### student name: yuzhi Lu
### student id: s4547009
=====================================
# File descriptions and use guide
## 1 Final version files
### unzip.py
Unzip the raw dataset and save the left and right knee to different directory.
(Where, ‘your_ Path’ refers to the file path of your dataset (ZIP format, do not unzip))

### split.py
Preprocess the image (decode and resize), shuffle and split into training and test set. 

### model_create.py
Use Resnet50_ V2(tf library) as my base, define a network, adjust parameters for training.

### main.py
Calling and compile process.


### test.ipynb
Model training and testing, including programming and visualization of the code. The precision and loss of 4 test set:
The accuracy and loss of the model in the test set are 0.9990363121032715 and 0.003378412453457713 respectively.

### Record_ training_ log.ipynb
Record the training log.

## 2 Intermediate files (These files are only used to show the interim results and not required for the final version)
### First version.ipynb 
This is the first version of my project including data unzip and split process.

### MobileNettest.ipynb
This file use MobileNet for transfer learning. However, the accuracy cannot fit our need.

## 3 visualization
Effect picture on training set and test set:
![](https://github.com/lyzAlbion/AKOA-knee/blob/main/output.png)

