# Brain MRI Super-Resolution Network
The model implemented is able to reconstructs a high-resolution version of an image from a low-resolution (factor of 4) version given.\
The working principles were to train a customized CNN network with tuple of (low-res image, high-res image) and optimize mean squared error between prediction and high-res image.\
Peak signal-to-noise ratio (PSNR) is commonly used to quantify reconstruction quality for images and is defined via MSE.
Lower MSE indicates higher PSNR and reconstruction quality.

The basic structure was inspired by ESPCN (Efficient Sub-Pixel CNN) proposed by Shi, 2016 and Keras implementation by Long, 2020. The model was also embedded with Residual Dense Blocks, inspired by Chakraborty, 2021. Residual blocks prevented early saturation or degradation with accuracy from increasing number of layers.

ADNI MRI Dataset were used in this work (see citation)
## Dependencies and Versions

- Ubuntu 16.04 or higher (Ubuntu 22.04.1 LTS was in used)
- NVIDIA® GPU drivers version 450.80.02 or higher (515.76 was in used)

| Package | Version |
| --- | --- |
| Graphic Card | GTX 3080 |
| System | Ubuntu 22.04.1 LTS|
| NVIDIA® GPU drivers | 515.76 |
| CUDA Toolkit | 11.2 |
| cuDNN SDK | 8.1.0 |
| Python | 3.9.13 |
| Numpy | 1.21.5 |
| Keras | 1.1.2 |
| Pillow | 2.9.1 |
| matplotlib | 3.5.2 |
| Tensorflow | 2.9.1 |

## Example Input and Output
The original image before down-sampling by factor of 4:\
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Original.png?raw=true)

The model input is:\
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Low_Res.png?raw=true)

The model output is:\
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Model_Prediction.png?raw=true)

PSNR of low resolution image and high resolution image is 25.5161
PSNR of predict and high resolution is 27.4348
PSNR improvement between low resolution and prediction 1.9188

## Model Summary
The model can be seen as:\
Input => Conv => Conv => Residual_Block(4 Conv) => Conv => Residual_Block(4 Conv) => Sub-pixel_Conv.\
As mentioned above Residual blocks prevented early saturation or degradation with accuracy from increasing number of layers.\
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Residual_Block.png?raw=true)

### Detail of model
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Model_summary.png?raw=true)

## Pre-processing
The dataset has 30520 samples of Alzheimer’s disease (AD) and Cognitive Normal (CN).
| Training set   | Validation set | Testing set    |
| -------------- | -------------- | -------------- |
| 17216          | 4304           | 9000           |

The dataset was resized to 300x300 and normalized from scale of (0,255) to (0,1).\
Both train_ds and validation_ds were assigned in form of **tuple** (low-res_ds, high-res_ds).

## Utility functions
These are some important utility functions, details are commented thoughout the code.

### dataset_preprocessing
Scale normalization, color space convertion and create tuple of dataset for trainning.
### setup_dataset
Create image dataset from directory with 8:2 ratio.
### scaling
Normalize an image from scale of (0,255) to (0,1).
### process_input/target
Convert rgb color spcae to yuv color space to extract luminance information.
### get_lowres_image
Utlize BICUBIC to downsample an image by a specified factor.
### upscale_image
Reconstruct a low-resolution image to a high-resolution through a model.
### plot_results
Plot results with additional zoom-in at a facotor of 4
### ESPCNCallback
The ESPCNCallback object will compute and display the PSNR metric during training and testing

## Reference
Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016). Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).\
Long, X. (2020). Image Super-Resolution using an Efficient Sub-Pixel CNN. https://keras.io/examples/vision/super_resolution_sub_pixel/#define-callbacks-to-monitor-training
\ADNI dataset for Alzheimer’s disease. (2022). ADNI MRI Dataset[Data set]. https://adni.loni.usc.edu/
