# MRI Super-Resolution Network on the ADNI Brain Dataset
***
This model takes upscaled (approximately by 4x) images from the ADNI Brain Dataset and downscales them.
***




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
