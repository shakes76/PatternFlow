# Brain MRI Super-Resolution Network
The model implemented is able to reconstructs a high-resolution version of an image from a low-resolution (factor of 4) version given. The basic structure was inspired by ESPCN (Efficient Sub-Pixel CNN) proposed by Shi, 2016 and Keras implementation by Long, 2020.

The model was also embedded with Residual Dense Blocks, inspired by Chakraborty, 2021. Residual blocks prevented early saturation or degradation with accuracy from increasing number of layers.

## Dependencies and Versions

- Ubuntu 16.04 or higher (Ubuntu 22.04.1 LTS was in used)
- NVIDIA® GPU drivers version 450.80.02 or higher (515.76 was in used)

| Package | Version |
| --- | --- |
| CUDA Toolkit | 11.2 |
| cuDNN SDK | 8.1.0 |
| Python | 3.9.13 |
| Numpy | 1.21.5 |
| Keras | 1.1.2 |
| Pillow | 2.9.1 |
| matplotlib | 3.5.2 |
| Tensorflow | 2.9.1 |

## Example Input and Output
The original image before down-sampling by factor of 4:
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Original.png?raw=true)

The model input is:
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Low_Res.png?raw=true)

The model output is:
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Model_Prediction.png?raw=true)

PSNR of low resolution image and high resolution image is 25.5161
PSNR of predict and high resolution is 27.4348
PSNR improvement between low resolution and prediction 1.9188

## Model Summary
The model can be simplified as:
Input => Conv => Conv => Residual_Block(4 Conv) => Conv => Residual_Block(4 Conv) => Sub-pixel_Conv
As mentioned above Residual blocks prevented early saturation or degradation with accuracy from increasing number of layers.
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Residual_Block.png?raw=true)

### Detail of model
![alt text](https://github.com/LingxiaoGao/PatternFlow/blob/topic-recognition/recognition/44708627_%20Efficient_Sub_Pixel_CNN/Demo_Example/Model_summary.png?raw=true)

## Pre-processing
The dataset has 30520 samples of Alzheimer’s disease (AD) and Cognitive Normal (CN).
| Training set   | Validation set | Testing set    |
| -------------- | -------------- | -------------- |
| 17216          | 4304           | 9000           |

The dataset was resized to 300x300 and normalized from scale of (0,255) to (0,1).
Both train_ds and validation_ds were assigned in form of **tuple** (low-res_ds, high-res_ds)
