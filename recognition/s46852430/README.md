# Segmentation of ISICs 2018 dataset with U-Net

# Algorithm
This is an improved Unet model which work on the basic Unet. Unet works on convolution, maxpooling and upsampling. This is all done by using Keras layer functions. There is a conv_block which is used to shorten the code as it acts like a shortcut, when the main model calls this function, it will perform 2 2D convolution, 2 normalization and also 2 activation. In the main model, we first downsample the image, and then upsample it. From downsampling to upsamplaing, this make the model to look like a U, thus the name. We use a batch number of 32, learning rate of 0.1 and a momentum of 0.9 to train the data. 
Apart from that, we use a 0.7 spilt for train set and 0.15 for test and valid dataset respectively. As for the metrics, we use a Dice similarity coeffient.

# Dependencies
Python 3.9.7
TensorFlow 2.5.0
matplotlib 3.4.3
ISICs 2018 Challenge dataset

# Model 
![model_plot](https://user-images.githubusercontent.com/56865527/197185218-dc1c81a0-1914-4328-955b-2c0e610299ec.png)

