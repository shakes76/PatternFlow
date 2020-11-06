# Image Segmentation on ISICs dataset using Improved Unet

## Backgournd
### Problem Statement
ISICs dataset contains images of Melanoma, a serious form of skin cancer. The aim of this project is to conduct image segmentation, creating pixel-wise binary mask for Melanoma. The dataset contains 2594 photos of Melanoma and 2594 masks. 
Dataset can be downloaded from https://challenge2018.isic-archive.com/

### Algorithm
U-Net is a convolutional network architecture, designed for biomedical image segmentation. It consists of one contraction part, bottleneck and expansion part. Any contraction block and expansion block at the same level will be concatenated. (Ronneberger, Fischer, Brox, 2015)
<br /><br /><a href="url"><img src="images/unet.png" height="300" width="600" ></a><br /><br />
We are using the improved U-Net, developed based on U-Net in 2017. The following features are added to normal U-Net:
1. Context modules for activations
2. Localization pathway to transfer features from lower level to higher spatial resolution
3. Segmentation layers integration at different levels for deep supervision
<br /><br /><a href="url"><img src="images/improved-unet.png" height="300" width="600" ></a><br /><br />

## Implementations
### Data Split
70% for training, 15% for validation, 15% for testing.

### Model Training Parameters
* Activation function: Sigmoid
* Optimizer: Adam (learning rate at 0.0001)
* Loss function: BinaryCrossEntrypy
* Checkpoint Callback: Maximum validation accuracy
* Training Epoch: 250

### Result
![Accuracy](images/acc1.png)
![Loss](images/loss1.png)
![dsc](images/dsc1.png)
![Predictions](images/predictions.png)<br />
(First row: row images; Second row: ground truth; Third row: predictions)

## Dependencies
* Python=3.7
* Tensorflow-gpu(2.1)
* Opencv
* Matplotlib
* PIL


## Author
Name: Wenjun Zhang
Student Number: s4469251
