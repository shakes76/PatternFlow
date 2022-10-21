# Image Segmentation of ISICs dataset with Unet


### Dependencies
* Tensorflow-gpu 2.1
* Matplotlib

### Description
The aim of the project is to successfully perform lesion segmentation on the ISIC dataset. Classifying each pixel either black or white, the following algorithm achieves this by using Unet. Images are resized so they are 512x512 in dimension. This particular size was picked because any smaller and completely black images would return a high dice coefficient which we don't want. 

The files are first extracted and then the data is split such that 80% of the dataset is used for training and 10% for both validation and test. A large percentage was used due to the fact there are only around 3K images in the dataset. Datasets for training, validation and testing are created such that it takes the tuple of all the images and masks files. The data is then shuffled and the filenames are mapped to data arrays. Masks are one-hot encoded and both images and masks are resized to 512x512. The model is then trained for 6 epochs using the dice coefficient as a metric. The Unet model was implemented following the article referenced below in References section.

Below are the results of the model before training compared to the ground truth, the decimal values corresponds to the dice coefficient for each mask starting from the left, as can be seen before training it is relatively low and has high variance.

![beforetrain](/resources/beforetrain.png)


### Training and Results
The model is trained over 6 epochs, where a dice coefficient of 0.79-0.80 is achieved over the validation data. 

Train for 65 steps, validate for 9 steps

Epoch 1/6
65/65 [==============================] - 120s 2s/step - loss: 0.5389 - dice_coef: 0.6231 - accuracy: 0.7553 - val_loss: 0.4502 - val_dice_coef: 0.7188 - val_accuracy: 0.7475

Epoch 2/6
65/65 [==============================] - 87s 1s/step - loss: 0.4029 - dice_coef: 0.7321 - accuracy: 0.8077 - val_loss: 0.3928 - val_dice_coef: 0.7148 - val_accuracy: 0.7479

Epoch 3/6
65/65 [==============================] - 137s 2s/step - loss: 0.3638 - dice_coef: 0.7583 - accuracy: 0.8077 - val_loss: 0.3363 - val_dice_coef: 0.7822 - val_accuracy: 0.7479

Epoch 4/6
65/65 [==============================] - 68s 1s/step - loss: 0.3611 - dice_coef: 0.7643 - accuracy: 0.8077 - val_loss: 0.3308 - val_dice_coef: 0.7581 - val_accuracy: 0.7479

Epoch 5/6
65/65 [==============================] - 128s 2s/step - loss: 0.3421 - dice_coef: 0.7753 - accuracy: 0.8395 - val_loss: 0.3456 - val_dice_coef: 0.7562 - val_accuracy: 0.8724

Epoch 6/6
65/65 [==============================] - 97s 1s/step - loss: 0.3169 - dice_coef: 0.7933 - accuracy: 0.8823 - val_loss: 0.3091 - val_dice_coef: 0.7801 - val_accuracy: 0.9001

Here below the predictions on the test data, we can see that we achieve better results than before training with some dice coefficients being in the 0.9 values.

![training](/resources/predictions.png)

The average dice coefficient when using the model to predict the mask on all the test data was calculated and the average was around 0.8.

#### Plot of the graph
The results of the dice coefficient over 6 epochs


![results](/resources/plot.png)

#### References
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
U-Net: Convolutional Networks for Biomedical Image Segmentation :Olaf Ronneberger, Philipp Fischer, Thomas Brox
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015