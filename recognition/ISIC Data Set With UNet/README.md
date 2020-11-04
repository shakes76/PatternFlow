# Segmentation of ISIC data set with U-Net
This is my solution to the ISIC data set using a U-Net model. The aim of this solution is to train a convolutional neural network to segment the ISIC data, which is a collection of photographs of skin lesions. A U-Net model was used as the convolutional neural network and is a great choice for this type of segmentation problem. In the end, I was able to make predictions with a dice similarity coefficient of 0.79.

An example prediction made using the model is shown below. From left to right we have: the original skin lesion image, the correct segmentation, my model's predicted segmentation.

<p align="center">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20165549.png">
<p align="center">Prediction made using U-Net model.</p>
</p>

I will now outline the contents of this repository, before discussing how the model was trained and whether it can made useful predictions for this type of problem.

## Contents of this Repository
The contents of this repository are detailed below.

| Scripts | Folders |
| ------- | ------- |
| `driver_script.py` `solution.py` | `Images` |

`driver_script.py` imports the ISIC data, creates, compiles and trains the U-Net model, and analyses the performance of this model.

`solution.py` contains the actual U-Net model used by driver_script.py.

This `Images` folder contains a number of images relating to the training, predicting and analysing of the model. Many of these images are included in this README.md file. Feel free to view the images in this folder to explore more examples of the training data, predictions made and training history.

## solution.py
This file is used to create the U-Net model. It is implemented entirely in TensorFlow. This file does not need to be run. Instead, it is imported into driver_script.py

| Dependencies | Functions |
| --- | --- |
| `tensorflow` `tensorflow.keras.layers` | `unet_model` |

Below I have detailed each of the functions in this script, describing their purpose.

`model` creates the U-Net model using the structure specified.

## driver_script.py
This is the driver script. This file imports the data, manipulates the data into various datasets for training, validating and testing, imports the model from solution.py and compiles this model, trains the model using the datasets, makes and plots predictions using the model.

| Dependencies | Classes (and their methods) | Functions |
| --- | --- |
| `tensorflow` `matplotlib.pyplot` `math` `glob` `IPython.display.clear_output` `tensorflow.keras.backend` | `DisplayCallback` `on_epoch_end` | `import_ISIC_data` `process_path` `decode_jpg` `decode_png` `analyse_training_history` `display_predictions` `display_data` `display` `compute_dice_coefficients` `dice_coefficient_loss` `dice_coefficient` |

Below I have detailed each of the functions in this script, describing their purpose.

`import_ISIC_data()` downloads the ISIC dataset from a specified location. Manipulates the data into training, validating and testind datasets.

`process_path(image_fp, mask_fp)` processes an image and a mask by decoding and normalising them.

`decode_jpg(file_path)` decodes and resizes a jpeg image.

`decode_png(file_path)` decodes and resizes a png image.

`analyse_training_history(history)` plots the acuraccy and validation accuracy of the model as it trains.

`display_predictions(model, ds, n=1)` makes n predictions using the model and the given dataset and displays these predictions.

`display_data(ds, n=1)` displays n images and masks from a given dataset.

`display(display_list)` displays plots of the provided data.
 
`compute_dice_coefficients(model, ds)` computes the average dice similarity coefficient for all predictions made using the provided dataset.

`dice_coefficient_loss(y_true, y_pred)` computes the dice similarity coefficient loss for a prediction.

`dice_coefficient(y_true, y_pred, smooth = 0.)` computes the dice similarity coefficient for a prediction.

## How to Run the File
In order to train this model as I have, follow the next steps:
* You will firstly need to download the ISIC data locally.
* Then, you will need to edit `driver_script.py` so that the images and masks are imported correctly.
* Then, you will need to make sure that all the dependencies listed previously are present in the environment.
* Then, you will need to simply run `driver_script.py`, and do not require any commandline arguments.

## Results
Below is an outline of how I used the model to segment the ISIC data set using the UNet model, alongside some results and images.

### Plotting an Example Image
When we run the driver script, the first thing it does it output an example image and mask from the training dataset. This image is shown below.

<p align="center">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Example%20Images/Figure%202020-11-03%20162140%20(0).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Example%20Images/Figure%202020-11-03%20165055%20(1).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Example%20Images/Figure%202020-11-03%20165055%20(2).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Example%20Images/Figure%202020-11-03%20165055%20(4).png">
<p align="center">Example Images from the training dataset</p>
</p>

### Creating the Model & Outputting Structure
Next, the model is created. The model summary is output, which I have included below. Note that the below output has been simplified.

| Layer (type)   | Output Shape    | Param #  | Connected to              |
| -------------- | --------------- | -------- | ------------------------- |
| InputLayer 1   | (256, 256, 1)   | 0        |                           |
| Conv2D 1       | (256, 256, 6)   | 60       | InputLayer 1              |
| Conv2D 2       | (256, 256, 6)   | 330      | Conv2D 1                  |
| MaxPooling2D 1 | (128, 128, 6)   | 0        | Conv2D 2                  |
| Conv2D 3       | (128, 128, 12)  | 660      | MaxPooling2D 1            |
| Conv2D 4       | (128, 128, 12)  | 1308     | Conv2D 3                  |
| MaxPooling2D 2 | (64, 64, 12)    | 0        | Conv2D 4                  |
| Conv2D 5       | (64, 64, 24)    | 2616     | MaxPooling2D 2            |
| Conv2D 6       | (64, 64, 24)    | 5208     | Conv2D 5                  |
| MaxPooling2D 3 | (32, 32, 24)    | 0        | Conv2D 6                  |
| Conv2D  7      | (32, 32, 48)    | 10416    | MaxPooling2D 3            |
| Conv2D  8      | (32, 32, 48)    | 20784    | Conv2D 7                  |
| MaxPooling2D 4 | (16, 16, 48)    | 0        | Conv2D 8                  |
| Conv2D 9       | (16, 16, 96)    | 41568    | MaxPooling2D 4            |
| Conv2D 10      | (16, 16, 96)    | 83040    | Conv2D 9                  |
| UpSampling2D 1 | (32, 32, 96)    | 0        | Conv2D 10                 |
| Concatenate 1  | (32, 32, 144)   | 0        | UpSampling2D 1 & Conv2D 8 |
| Conv2D 11      | (32, 32, 48)    | 62256    | Concatenate 1             |
| Conv2D 12      | (32, 32, 48)    | 20784    | Conv2D 11                 |
| UpSampling2D 2 | (64, 64, 48)    | 0        | Conv2D 12                 |
| Concatenate 2  | (64, 64, 72)    | 0        | UpSampling2D 2 & Conv2D 6 |
| Conv2D 13      | (64, 64, 24)    | 15576    | Concatenate 2             |
| Conv2D 14      | (64, 64, 24)    | 5208     | Conv2D 13                 |
| UpSampling2D 3 | (128, 128, 24)  | 0        | Conv2D 14                 |
| Concatenate 3  | (128, 128, 36)  | 0        | UpSampling2D 3 & Conv2D 4 |
| Conv2D 15      | (128, 128, 12)  | 3900     | Concatenate 3             |
| Conv2D 16      | (128, 128, 12)  | 1308     | Conv2D 15                 |
| UpSampling2D 4 | (256, 256, 12)  | 0        | Conv2D 16                 |
| Concatenate 4  | (256, 256, 18)  | 0        | UpSampling2D 4 & Conv2D 2 |
| Conv2D 17      | (256, 256, 6)   | 978      | Concatenate 4             |
| Conv2D 18      | (256, 256, 6)   | 330      | Conv2D 17                 |
| Conv2D 19      | (256, 256, 1)   | 7        | Conv2D 18                 |

In total, there are 276,337 parameters, all of which are trainable.

### Compiling the Model
Next, I compiled the model. 

**Optimizer**: I used the default Adam optimizer and did not change the learning rate. I found that this was an adequate optimizer and did not with to explore alternatives.

**Loss Function**: I used binary crossentropy as the loss function. The only alternative I considered here was to use the dice coefficient loss as a custom loss function. I decided not to in the end, due to the dice coefficient loss not giving me my desired convergence.

**Metrics**: I used accuracy as a metric. I have also used dice coefficient as a metric but chose accuracy in the end. Either of these would have sufficed here though.

### Training the Model
Next, I trained the model.

I trained the model over 100 epochs. I stopped at 100 epochs because I was getting good convergence and my training accuracy and validation accuracy.

I used a training and validating batch size of 32. This size sufficed and given my hardware felt this was more than plenty.

### Analysing the Training History
After training was complete, I analysed the training history by ploting how the training accuracy and validation accuracy changed over each epoch. The plot of the training history is shown below.

<p align="center">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Training%20History/Training%20History.png">
</p>
<p align="center">Training history of model after each epoch</p>

We can clearly see the training accuracy and validation accuracy converge over time. However, it is surprising to see this accuracy be so high for early epochs. This is where I believe that using the dice coefficient as a metric would have been more useful. This is because the dice coefficient can be a better measure of how accurate the model truly is performing, and we would see it have a far lower dice coefficient for these early epochs. An example image is shown below from an alternate model (trained using the same datasets, optimizer and loss function).

<p align="center">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Training%20History/Training%20History%20Other%20Model.png">
</p>
<p align="center">Training history of model after each epoch</p>

Clearly in this model, we can see how the dice coefficient improved slowly but surely over time.

### Making Predictions
Next, I made and displayed some predictions using the model to test that it is accurate. Five of these such predictions are shown below.

<p align="center">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20165549.png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20164859%20(10).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20164859%20(13).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20164859%20(9).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20165531.png">
<p align="center">Good predictions made using the model</p>
</p>

As you can see, these are all really good predictions, and in my opinion are very usable. The first four in particular are highly accurate. One shortcoming I found was how my model dealt with the edges of the segmentation. As you can see in the fifth prediction, my prediction does not find the correct outline that was needed.

However, not all predictions are this good. For example, take a look at the following two predictions.

<p align="center">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20162140%20(102).png">
  <img src="https://github.com/maxhornigold/PatternFlow/blob/topic-recognition/recognition/ISIC%20Data%20Set%20With%20UNet/Images/Prediction%20Images/Figure%202020-11-03%20164859%20(15).png">
<p align="center">Poor predictions made using the model</p>
</p>

These are clearly not usable predictions. I found that often the model got tricked up by certain skin features. If we look at the first prediction, the four black dots trick the model, which completely misses the actual lesion. On the other hand, in the second prediction, the skin lesion is correcly identified, however the model also incorrectly segments many other parts of the image which are not skin lesions. These two examples, which are not compeltely uncommon in the predictions made, show that the model still needs a lot of work if it is going to be used on all images, particularly trickier images which are obscured.

### Analysing Dice Similarity Coefficient of Model
Next, I computed the dice similarity coefficient. I did this by recording the dice similarity coefficient for each prediction made using the testing data. Then, I average these values to find the average dice similarity coefficient. Doing this, I found that the average dice similarity coefficient was 0.79369396. Overall, this is a reasonably good average. However, being an average, we do lose the importance of how innacurate the model can be in some tricky images.

## Improvements to the Model?
Whilst this model meets the 0.7 target accuracy for the dice similarity coefficient, clearly I have shown there is still great room for growth and construct a superior model which does not have the same shortcomings. Some potential areas to explore here are:

* Using dice coefficient loss as a loss function, as opposed to binary crossentropy.
* Incorporating dropoutlayers into the model to reduce overfitting.
* Increasing the number of filters in the model.
* Increasing the batch size and number of epochs when training the model.
* Varying the learning rate.

It is very likely that by simply incorporating some of these would result in a superior model.

If I wanted to go even further, I could implement a modified U-Net, which is simply an expansion of the current vanilla U-Net used in this model. A modified U-Net will almost certainly allow me to reach higher accuracy levels.
