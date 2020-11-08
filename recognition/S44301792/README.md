# Image Classification
This algorithm Model.ipynb is tried to classify laterality (left or right sided knee) of the OAI AKOA knee data set having a minimum accuracy
of 0.9 on the test set. 

### Introduction
At the the start, the Algorithm get the data from AKOA_Analysis Data set And Separated into Three data sets which are 
* Training Data 
* Testing Data 
* Validation Data

Also the Algorihtm converting the image channels from Three channels into one channel. The examples of images after change are labeled below
![Image of Exmaples](https://github.com/theHughJin/PatternFlow/blob/master/recognition/S44301792/Image/Screen%20Shot%202020-11-07%20at%203.36.20%20PM.png)
        
This algorithm Model.ipynb is tried to classify laterality (left or right sided knee) of the OAI AKOA knee data set having a minimum accuracy
of 0.9 on the test set. 

The structure of the model is 
*  A 16 filters Convolutional Layers with relu as the activation function 
*  A 32 filters Convolutional Layers with relu as the activation function 
*  A flatten Layer
*  A Dense layer with SoftMax function as output layer
*The diagram below is the Summary of model
        ![Image of Summary of Model](https://github.com/theHughJin/PatternFlow/blob/master/recognition/S44301792/Image/Screen%20Shot%202020-11-08%20at%2011.54.59%20AM.png)
### Prerequest
Tensorflow V2.0+ and Python V3.5+. Download the Glob and matplotlib library. Download Tensorflow [here](https://www.tensorflow.org/install)


