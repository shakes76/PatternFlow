# Segmentation of ISICs 2018 dataset with U-Net

COMP3710 Report Task 3

Teguh Salim (45018332)

## Objective

This program aims to solve a segmentation problem, which is to classify pixels in an image by groups. In this case, the problem is to identify lesion vs healthy skin and there are only 2 groups, also known as binary segmentation problem.

## Approach
A U-Net model as described in (Ronneberger, et al., 2015). The model described in the paper was implemented exactly as described by recreating convolution, maxpool, and upsampling layers using Keras layers functions. 

Adam optimizer was used with learning rate of 0.0001 which was found to yield satisfactory results with less epoch. The metric used to evaluate the results is Dice Similarity Coefficient (DSC) which measures similarity between two sets of data, in this case by comparing between the predicted images from training model and the ground truth masks provided in the dataset. The DSC is implemented using Keras functions to apply the following equations to tensors: (2 * |X ∩ Y|) / (|X| + |Y|).

The U-Net model consists of downsampling part for feature learning and upsampling part for mask segmentation, each part consists of 4 convolution layers, the result of each layer in the downsampling part is also concatenated to the corresponding layer in the upsampling part before convolution, achieving better segmentation performance.

## Parameters and results
The input images, which originally has varying sizes were resized to (128,128) for faster training. The training data was split between training:validation:testing on 70:20:10 ratio, which seems to be a commonly prescribed starting point for smaller datasets. Batch size of 8 and Adam optimizer learning rate of 0.00001 was used, which gives satisfactory results, a validation DSC of 0.7681 was obtained after 5 epochs and subsequently the trained model gives a DSC of 0.7560 on the test dataset. The results outputs are given below.

Training results:
Train for 226 steps, validate for 64 steps
Epoch 1/5
226/226 [==============================] - 100s 444ms/step - loss: 0.6417 - dsc: 0.3583 - val_loss: 0.4823 - val_dsc: 0.5177
Epoch 2/5
226/226 [==============================] - 98s 434ms/step - loss: 0.4169 - dsc: 0.5831 - val_loss: 0.3622 - val_dsc: 0.6378
Epoch 3/5
226/226 [==============================] - 94s 415ms/step - loss: 0.3797 - dsc: 0.6203 - val_loss: 0.3218 - val_dsc: 0.6782
Epoch 4/5
226/226 [==============================] - 93s 410ms/step - loss: 0.2873 - dsc: 0.7127 - val_loss: 0.2695 - val_dsc: 0.7305
Epoch 5/5
226/226 [==============================] - 100s 443ms/step - loss: 0.2385 - dsc: 0.7615 - val_loss: 0.2319 - val_dsc: 0.7681

Testing results:
32/32 [==============================] - 10s 304ms/step - loss: 0.2440 - dsc: 0.7560
[0.24397906474769115, 0.756021]

The following plot was also obtained:
![Alt text](plots/graph.png?raw=true "Training results")

## Program structure
The program has 5 modules: processdata.py processed the raw data and rearrange the folder structure into training, test, and validation sets for easier processing, imagegen.py creates image generator using Keras ImageDataGenerator function on the processed folder to obtain the data from images in batches for training, unet.py creates the U-Net model and dice.py calculates DSC and Dice loss. The driver script is main.py which runs all the other modules and obtain the results.

## Assumption and dependencies
The user is assumed to have downloaded and unzipped the ISICs 2018 dataset to the root directory as is. This folder is not commited to repo due to its size and is assumed to be available locally.

The program was tested with the following dependencies:
* Python 3.7.9
* TensorFlow 2.1
* matplotlib
* glob, shutil

## References
O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015. University of Freiburg. Springer. Obtained from: https://arxiv.org/abs/1505.04597.