# Improved UNet model for segmentation of OASIS brain data set

## Problem description

The OASIS brain data set needs to be segmented in order to accurately identify the changes that occur in brain structure. For the purpose of performing this segmentation, deep learning is a widely used method that yields highly significant results from a performance perspective. In particular, the UNet model is used extensively to segment the brain data set. The problem aims at developing an improved UNet model that generates a minimum Dice similarity coefficient of 0.9 for all labels on the test set.

## Algorithm

The train data set contains 9664 images 256x256 images. Data augmentation is performed on this data using the ImageDataGenerator module of Tensorflow. A generator object is created for the training data set using the segmented training data as the mask.

The UNet architecture consists of a successively contracting path followed by a succesively expanding path. The contracting path consists of repeated usage of 2D convolutional layers, with each using a rectified linear unit as the activation function and then followed by a max pooling operation with pool size of 2x2, while the number of filters gets increased from 64 to 128 to 256 to 512 to 1024 in each step. To prevent overfitting, dropout with rate 0.5 is also applied to the convolutional layers with 512 and 1024 filters. The contracting path's objective is to increase the feature information while decreasing the spatial information. The expanding path then combines the spatial and feature information by repeatedly applying a 2D convolutional layer on an upsampling layer of size 2x2, concatenating it with the corresponding feature in the contracting path, and subsequently applying another 2D convolutional layer. The number o filters is decreased in each step to 512 to 256 to 128 to 64 to 2. A final convolutional layer with sigmoid activation function is applied to generate the final output.

The model is created using an input of 256x256 with batch size 1 and using the Adam optimizer with a learning rate of 0.0001. The dice similarity coefficient is used as the metric while the loss function is binary cross-entropy loss. The model is fit on the train generator object and the number of epochs is set to 5 with 2000 steps in each epoch.

To predict the output on the test set, the test data is augmented using the ImageDataGenerator module and passed to the model. The results are stored as PNG files in the **results\\** directory. Evaluation of the model on the validation data set produces a dice similarity coefficient of 0.9207.

## Instructions to setup
1. Download the preprocessed OASIS brain dataset to the **recognition\\calvin** directory.
2. Inside the directory containing the brain data set, create an empty directory and name it **results\\**.
2. From the **recognition\\calvin** directory, run the command
```pip3 install - requirements.txt```
to install the dependencies (tensorflow, skimage).

## Instructions to run
From the **recognition\\calvin** directory, run the command
```py example_driver.py```
to execute the driver script.
