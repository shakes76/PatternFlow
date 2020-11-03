# Segmentation of ISIC data set with U-Net
This is my solution to the ISIC data set using a U-Net model. The aim of this solution is to train a convolutional neural network to segment the ISIC data, which is a collection of photographs of skin lesions. A U-Net model was used as the convolutional neural network and is a great choice for this type of segmentation problem. In the end, I was able to make predictions with a dice similarity coefficient of 0.7.

An example prediction is shown below. From left to right we have: the original skin lesion image, the correct segmentation, my model's predicted segmentation.

![alt text]("")

I will now outline the contents of this repository, before discussing how the model was trained and whether it can made useful predictions for this type of problem.

## Contents of this Repository
This repository contains  contains the following two files:
* driver_script.py
* solution.py

driver_script.py imports the data, creates, compiles and trains the U-Net model, and analyses the performance of this model.

solution.py contains the actual U-Net model used by driver_script.py.

The folder also contains the folder:
* Images

This folder contains a number of images relating to the training, predicting and analysing of the model. Many of these images are included in this README.md file.

## solution.py
This is the U-Net model. It is implemented entirely in TensorFlow. This file does not need to be run. Instead, it is imported into driver_script.py

## driver_script.py
This is the driver script. This file:
* imports the data.
* manipulates the data into various datasets for training, validating and testing.
* import the model from solution.py and compile this model.
* train the model using the datasets.
* makes and plots predictions using the model.

Includes a number of functions:
* import_ISIC_data()
* process_path()
* decode_jpg()
* decode_png()
* analyse_training_history()
* display_predictions()
* display_data()
* display()
* compute_dice_coefficients()
* dice_coefficient_loss()
* dice_coefficient()

## How to Run the File
In order to train this model as I have, follow the next steps:
* You will firstly need to download the ISIC data locally.
* Then, you will need to edit the driver_script so that the images and masks are imported correctly.
* Then, you will need to make sure that all the dependencies listed previously are present in the environment.
* Then, you will need to simply run the dripver_script.py file.

## Results
Below is an outline of how I used the model to segment the ISIC data set using the UNet model, alongside some results and images.

### Plotting an Example Image
When we run the driver script, the first thing it does it output an example image and mask from the training dataset.
This image is shown below.

![alt text](https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Example.png)

### Creating the Model & Outputting Structure

The model summary is output. The output is shown below.

Layer (type)                    Output Shape         Param #     Connected to
input_2 (InputLayer)            [(None, 256, 256, 1) 0

conv2d_19 (Conv2D)              (None, 256, 256, 6)  60          input_2[0][0]

conv2d_20 (Conv2D)              (None, 256, 256, 6)  330         conv2d_19[0][0]

max_pooling2d_4 (MaxPooling2D)  (None, 128, 128, 6)  0           conv2d_20[0][0]

conv2d_21 (Conv2D)              (None, 128, 128, 12) 660         max_pooling2d_4[0][0]

conv2d_22 (Conv2D)              (None, 128, 128, 12) 1308        conv2d_21[0][0]

max_pooling2d_5 (MaxPooling2D)  (None, 64, 64, 12)   0           conv2d_22[0][0]

conv2d_23 (Conv2D)              (None, 64, 64, 24)   2616        max_pooling2d_5[0][0]

conv2d_24 (Conv2D)              (None, 64, 64, 24)   5208        conv2d_23[0][0]

max_pooling2d_6 (MaxPooling2D)  (None, 32, 32, 24)   0           conv2d_24[0][0]

conv2d_25 (Conv2D)              (None, 32, 32, 48)   10416       max_pooling2d_6[0][0]

conv2d_26 (Conv2D)              (None, 32, 32, 48)   20784       conv2d_25[0][0]

max_pooling2d_7 (MaxPooling2D)  (None, 16, 16, 48)   0           conv2d_26[0][0]

conv2d_27 (Conv2D)              (None, 16, 16, 96)   41568       max_pooling2d_7[0][0]

conv2d_28 (Conv2D)              (None, 16, 16, 96)   83040       conv2d_27[0][0]

up_sampling2d_4 (UpSampling2D)  (None, 32, 32, 96)   0           conv2d_28[0][0]

concatenate_4 (Concatenate)     (None, 32, 32, 144)  0           up_sampling2d_4[0][0]            
                                                                 conv2d_26[0][0]                                                           
conv2d_29 (Conv2D)              (None, 32, 32, 48)   62256       concatenate_4[0][0]              
conv2d_30 (Conv2D)              (None, 32, 32, 48)   20784       conv2d_29[0][0]

up_sampling2d_5 (UpSampling2D)  (None, 64, 64, 48)   0           conv2d_30[0][0]

concatenate_5 (Concatenate)     (None, 64, 64, 72)   0           up_sampling2d_5[0][0]            
                                                                 conv2d_24[0][0]
                                                                 
conv2d_31 (Conv2D)              (None, 64, 64, 24)   15576       concatenate_5[0][0]

conv2d_32 (Conv2D)              (None, 64, 64, 24)   5208        conv2d_31[0][0]       

up_sampling2d_6 (UpSampling2D)  (None, 128, 128, 24) 0           conv2d_32[0][0]        

concatenate_6 (Concatenate)     (None, 128, 128, 36) 0           up_sampling2d_6[0][0]            
                                                                 conv2d_22[0][0]                  

conv2d_33 (Conv2D)              (None, 128, 128, 12) 3900        concatenate_6[0][0]    

conv2d_34 (Conv2D)              (None, 128, 128, 12) 1308        conv2d_33[0][0]                  

up_sampling2d_7 (UpSampling2D)  (None, 256, 256, 12) 0           conv2d_34[0][0]                  

concatenate_7 (Concatenate)     (None, 256, 256, 18) 0           up_sampling2d_7[0][0]            
                                                                 conv2d_20[0][0]  
                                                                 
conv2d_35 (Conv2D)              (None, 256, 256, 6)  978         concatenate_7[0][0]   

conv2d_36 (Conv2D)              (None, 256, 256, 6)  330         conv2d_35[0][0]    

conv2d_37 (Conv2D)              (None, 256, 256, 1)  7           conv2d_36[0][0]                  

Total params: 276,337
Trainable params: 276,337
Non-trainable params: 0

### Compiling the Model
I used the adam optimizer.

I used binary crossentropy as the loss function.

I used dice coefficient loss as a metric.

### Training the Model
I used 50 epochs.

I used a training and validating batch size of 32.

### Analysing the Training History
After training was complete, I analysed the training history by ploting how the training dice coefficient and validating dice coefficients changed over each epoch. The plot of the training history is shown below. We can clearly see the training dice coefficient and validation dice coefficient converge over time.

![alt text](https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Training%20History.png)

### Making Predictions
Next, I displayed some predictions I made. Three such prediciton are shown below.

![alt text](https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%203.png)

![alt text](https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%204.png)

![alt text](https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%206.png)

As you can see, these are all really good predictions, and in my opinion are very usable. The dice similarity coefficients for these predictions are: _____, ____, and _____. This supports the accuracy of this prediction.

However, not all predictions are this good. For example, take a look at the following prediction.

![alt text]("")

The dice similarity coefficient of this prediciton is _____. This is clearly not a usable prediction.

### Analysing Dice Similarity Coefficient of Model
Next, I computed the dice similarity coefficient. I did this by recording the dice similarity coefficient for each prediction made using the testing data. Then, I average these values to find the average dice similarity coefficient. Doing this, I found that the average dice similarity coefficient was ___________.
