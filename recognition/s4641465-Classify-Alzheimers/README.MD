# MRI Super-Resolution Network on the ADNI Brain Dataset

## Background on ADNI Brain Dataset
The ![ADNI Brain Dataset](http://adni.loni.usc.edu/) used to train this model can be found preprocessed on the COMP3710 blackboard
***

This model takes upscaled (approximately by 4x) images from the ADNI Brain Dataset and downscales them.
***
Using several (in this case 3) convolutional layers to create a feature map, the model then creates a 16 layer sub-pixel convolutional layer. Using pixel shuffling these 16 layers are reconstructed into one layer where each pixel in the sub-pixel layer corresponds to a pixel in a 4x4 section of the greater high-resolution output layer, creating a higher-resolution image
![graph](images/graph.png)
***
Here are the results, high resolution images are taken from the test set and upscaled by 4x creating the low resolution image. These low res images are then input into the model which outputs the prediction.

![highres](images/0-highres.png)
![lowres](images/0-lowres.png)
![prediction](images/0-prediction.png)

The prediction is definitely an improvement on the low resolution image, but is clearly not as detailed as the high resolution image.

***
How to use:
- Add dataset images to new folder named "datasets"
- run train.py
- run predict.py



***
References:

https://keras.io/examples/vision/super_resolution_sub_pixel/
https://medium.com/@zhuocen93/an-overview-of-espcn-an-efficient-sub-pixel-convolutional-neural-network-b76d0a6c875e
