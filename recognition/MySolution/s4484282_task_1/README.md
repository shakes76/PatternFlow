## s4484282 Unet improved COMP3710

Segment the ISIC's dataset with improved UNet with all labels having a minimum
Dice similarity coefficient of 0.8 on the test set.

Sounds simple enough, but first, let's build a normal Unet.

# Testing the waters

Before endevouring to improve Unet, we must first understand what unet is.

Unet is a strategy for building convolutional neural networks (CNNs) with more data.

The main idea is through progressive pooling down to a specified point, and then upsampling and combining in a step-up manner, one can increase the resolution of the output.
An important modification in Unets over conventional CNNs in general is the large number of feature channels in the upsampling stage.
This allows for context information to propagate up the network to the higher resolution layers (which ultimately improves predictive accuracy).

As a consequence, the expansive path (the upsampling part) is more or less symmetrical to the contracting path (the pooling part). This is also why the method forms a 'U' shape and is why it's called a 'Unet'.

# [Improving the system](https://arxiv.org/pdf/1802.10508v1.pdf)

We can improve the original Unet model by introducing residual connections, which yeild minor improvements.
The Changes made are based off the 2017 BRATS challenge report listed above.

Taken from the report as well, the model we wish to implement is as follows.

![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/thing_we_implemented.PNG)

This model, as stated previously, improves the accuracy (slightly) of the original Unet through the addition of small element wise summing of the convolutional layers at various stages of the contracting path.

# Testing accuracy
Accuracy was determined by largely using the [Dice loss coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). This was done via the `dice_coef()` and `dice_coeff_loss()` methods which were included as metrics in both the improved and regular models. 

# Colab document

The report and a large percentage of the code used is documented in the [Colab document](https://colab.research.google.com/drive/1YwaXD-fa3LNqCvG4Pb1gB-krDDrrUhF-?usp=sharing) (which was also used to train the model).

# Test scripts

The model can be tested by cloning the repository and running the `main.py` file in the `s4484282_task_1` directory. As per request, the model files have been removed (so no pre-trained model will be present). Hence, please ensure the
model is trained and saved as a `'h5` file on your local drive before you run the test functions.

Note that this model was largely developed and tested using google Colab. As such, you can verify the originality of the document, the integrity of the training and testing outcomes by viewing the [Colab document](https://colab.research.google.com/drive/1YwaXD-fa3LNqCvG4Pb1gB-krDDrrUhF-?usp=sharing) without the need to run the scripts. Note however that you will not have access my google drive, so you'll need to setup your own with the same training dataset offered by UQ.

# Regular Unet model output

![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/regular_out_1.PNG)
![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/regular_out_2.PNG)
![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/regular_out_3.PNG)

# Improved Unet model output
![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/Improved_out_1.PNG)
![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/Improved_out_2.PNG)
![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/Improved_out_3.PNG)

# Improved Unet Tensorboard results
The results for training both models was about the same, so only the improved version was added in this readme

![](https://github.com/Despicable-bee/PatternFlow/blob/s4484282_task_1_unet_improved/recognition/MySolution/s4484282_task_1/example_output/improved_tensorboard_out.PNG)

# Result

The results (which were gathered from observing the validation output during training) are as follows:

Regular Unet:
accuracy: 0.9854 - dice_coef: 0.9568 - val_loss: 0.3785 - val_accuracy: 0.9270 - val_dice_coef: 0.8269

Improved Unet:
accuracy: 0.9801 - dice_coef: 0.9377 - val_loss: 0.3218 - val_accuracy: 0.9317 - val_dice_coef: 0.8336

In terms of validation accuracy, the improved Unet is better than the original on average by approximately 1%.
This improvement is consistent with the 2017 BRATS challenge report in which the improved Unet version was based off.

As with many ML models, the training accuracy (~98%) is far higher than the validation accuracy (~92%).

However in both cases, the validation dice coefficient were above the 0.8 score as per requirement.

## [EXTRA CREDIT STUFF](https://colab.research.google.com/drive/1NYo8jk9Rxc8qzklIroveVWH7CO-22n_6?usp=sharing)

I didn't do the stylegan2 implementation from scratch, but I did implement the model and train it using Nvidia's official implementation in pytorch. Take a look, I've added some nice documentation on SLURM for anyone who wants it.

HEY EVERY      !
IT'S ME, SPAMTON G. SPAMTON!