# Implementation of Segmentation on the Brain MRI data set with Unet

For solving the problem, I used Keras with a Tensorflow backend. 

The Brian MRI data was a pre-processed version from the teaching team of COMP3710.

Finally, we used Unet to implement segmentation on this pre-processed data set and with some results.


## Preparing the data for training

Firstly, I reshaped the train and test image data to be 4 diementions. Afterwards, I normalized them to be fit for used.

Secondly, I normalized the labels of trian and test data and let them to be categorical.


## Unet structure

The Unet structure will look like this:


<image width="700" src="pic/Unet.png" />

## Constructing Unet

I constructing a Unet model with the input shape of train and test shape. 
Following the structure of Unet, I downsampling the input and then upsampling again. 
I randomly add Dropoup layers between the Unet to reducing overfitting.


## Fit the model with normal Unet

To fit the model, I compiled the model with the learning rate at 0.0001, set the loss as binary_crossentropy, and used the metrics of accuracy.

The result is as follow:

<image width="700" src="pic/modelfit.PNG" />

## Dice coefficient with nromal Unet

The results of dice coefficient for normal Unet are as follows:

<image width="700" src="pic/dice.PNG" />

## Improved Unet structure

The improved Unet structure will look like this:

<image width="700" src="pic/improved_Unet.PNG" />


## Fit the model with improved_Unet

To fit the model, I compiled the model with the learning rate at 0.0001, set the loss as binary_crossentropy, and used the metrics of accuracy.

The result is as follow:

<image width="700" src="pic/improved_modelfit.PNG" />

## Dice coefficient with improved Unet

The results of dice coefficient for improved Unet are as follows:

<image width="700" src="pic/improved_dice.PNG" />



