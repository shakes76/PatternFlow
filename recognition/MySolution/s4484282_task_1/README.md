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

As a consequence, the expansive path (the upsampling part) is more or less symmetrical to the contracting path (the pooling part).

# [Improving the system](https://arxiv.org/pdf/1802.10508v1.pdf)

We can improve the original Unet model by introducing residual connections, which yeild minor improvements.
The Changes made are based off the 2017 BRATS challenge report listed above.

# Testing accuracy
Accuracy was determined by largely using the [Dice loss coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).

# Colab document

The report and a large percentage of the code used is documented in the [Colab document](https://colab.research.google.com/drive/1YwaXD-fa3LNqCvG4Pb1gB-krDDrrUhF-#scrollTo=Cihc63yObw1s) (which was also used to train the model).

# Test scripts

The model can be tested by cloning the repository and running the `script name` in the `directory name` directory. The test script uses a pre-trained model, and selects a particular image 

# Test Script example output

here is some example output.

# Result

The results are as follows:

Standard Unet:
accuracy: 0.9854 - dice_coef: 0.9568 - val_loss: 0.3785 - val_accuracy: 0.9270 - val_dice_coef: 0.8269

Improved Unet:
accuracy: 0.9801 - dice_coef: 0.9377 - val_loss: 0.3218 - val_accuracy: 0.9317 - val_dice_coef: 0.8336

In terms of validation accuracy, the improved Unet is better than the original on average by approximately 1%.
This improvement is consistent with the 2017 BRATS challenge report in which the improved Unet version was based off.
