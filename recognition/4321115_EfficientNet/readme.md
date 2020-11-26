# EfficientNet for cifar-10 dataset

The EfficientNet architecture was introduced in 2019 with the concept of expansion-squeezing, which processes RGB channels individually using a globalConvolution, separating each channel, and augmenting the respective feature spaces. This is called "compound coefficient" optimisation, effectively avoiding tedious hyperparameter tuning over deepper networks, and allowing the network's breadth and depth to be automatically adjusted for given recognition tasks. These are called autML architectures, and were presented as high achieving recognition models (84% and above for classifcation).

## Model

The model is a classic CNN with the exception of:
1. a pre-optimised compound coefficient parameter dictionary which gives the best values for parameters such as number of repeats of the same block, the expansion/squeeze ratios, the kernel size, strides of the block.
2. The swish activation which returns a linear loss from the sigmoid function. More information can be found here(https://www.google.com/search?client=safari&rls=en&q=swish+activation&ie=UTF-8&oe=UTF-8).

Current tests over a 1500 epoch session yielded an 81.7% accuracy, with a steady linear loss drop, theoretically giving more space for accuracy increase. the Distance between test and validation accuracy remained within 10%.

This(https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html) link gives more information about this architecture.

## Structure

The following gives an overview of this repository:
1. model.py: contains the model architecture block implementation and swish activation definition
2. train.py: initiates data loading, processing, train/test splitting and building the model.
3. plot.py : optionally plots the loss saved after training (loss.csv)

## Loading data

The data is loaded directly in the model.py definition, but can be adjusted for other data sets for classification.


