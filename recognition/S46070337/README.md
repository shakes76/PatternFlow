# Project description:
Segment the ISICs data set with the Improved UNet [1] with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

## Dataset description:
The data set contains and the original skin patch image segmentation, used to detect melanoma (skin cancer) area. The depth of neural network in this subset of a data set is the purpose of the training, then split test image, in order to identify areas have melanoma. The images in the dataset are of different sizes and then modified to fit the model in the pre-processing stage.

## Improved Unet architecture
I splited the all these pictures into training set ,validation set and test set with the ratio of 70%, 15% and 15% respectively in order to improve the training performance. To meet the convolution layers requirements, I try to nornormalize the images resized the images' shape as 256 * 256 * 1 and 256 * 256 * 3. 

Below shows the architecture of the Improved Unet model:

<image width="700" src="Screenshots/Improved unet.png" />

The improved unet has two parts like the original unet model does, which is encoding and decoding. As for the difference, the improved unet model has a convolution layer and a context module which original unet model does not have. 
The context module contains 7 layers
##### Two Normalization Layers
##### Two Activation Layers "LeakyReLU"
##### Two Convolution Layers 
##### One Drop Layer with dropout ratio 0.3
As well the upsampling module can replace the convolution layers' transpose, the localization module can combine the features from concatenation and decrease the amount of features. Additionaly with segmentation layers of the model, it can perform segmentation for pictures well. Finally, the output layer is for store the processed image shapes and the activation function is 'sigmoid'.


## Algorithms

My directory contains a README.md , a directory 'Screenshots' contains several pictures of the outputs from the two python files, model.py and test.py:

Improved Unet model has been built in model.py. and test.py contains other functions and scrips for running the algorithm and results visualization.

## Evaluation
All the evaluation procedure based on epoch 10.
#### 1. Visualization of the prediction
##### 1.1 Ground Truth
#####     The image below is the ground truth image:
<image width="256" image height="256" src="Screenshots/Unet.png" />

##### 1.2 Prediction
#####     The image below is the ground truth image:
<image width="256" image height="256" src="Screenshots/Unet.png" />


##### 1.3 Accuracy and Loss
#####     The image below is the Accuracy and Loss during training:
<image width="700" src="Screenshots/evaluation_curve.png" />

