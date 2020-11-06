# Image Segmentation of ISICs dataset with Unet


### Dependencies
* Tensorflow-gpu 2.1
* Matplotlib

### Description
The aim of the project is to successfully perform lesion segmentation on the ISIC dataset. Classifying each pixel either black or white, the following algorithm achieves this by using Unet. Images are first resized so they are 512x512 in dimension. This particular size was picked because any smaller and completely black images would return a high dice coefficient which we don't want. The images are processed with one-hot encoding and the model is trained using Unet with the processed images as input. 

80% of the dataset was used for training and 10% for both validation and test. A large percentage was used due to the fact there are only around 3K images in the dataset.

Below are the results of the model before training compared to the ground truth, the decimal values corresponds to the dice coefficient for each mask starting from the left, as can be seen before training it is relatively low with values ranging from 0 to 0.2.
![beforetrain](/resources/beforetrain.png)

### Training and Results
The model is trained over 6 epochs, where a dice coefficient of 0.8153 is achieved over the validation data. 
Here below the predictions on the test data, we can see that we achieve better results than before training with some dice coefficients being in the 0.9 values.
![training](/resources/predictions.png)

The average dice coefficient when using the model to predict the mask on all the test data images is 0.82. 

#### Plot of the graph
The results of the dice coefficient over 6 epochs


![results](/resources/plot.png)

