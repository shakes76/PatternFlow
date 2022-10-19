# Brain MRI Super-Resolution Network
The model implemented is able to reconstructs a high-resolution version of an image from a low-resolution (factor of 4) version given. The basic structure was inspired by ESPCN (Efficient Sub-Pixel CNN) proposed by Shi, 2016 and Keras implementation by Long, 2020.

The model was also embedded with Residual Dense Blocks, inspired by Chakraborty, 2021. Residual blocks prevented early saturation or degradation with accuracy from increasing number of layers.

## Dependencies and Versions
Ubuntu 16.04 or higher (Ubuntu 22.04.1 LTS was in used)
NVIDIA® GPU drivers version 450.80.02 or higher (515.76 was in used)

CUDA Toolkit: 	 11.2.
cuDNN SDK:	 8.1.0.
Python:		 3.9.13
Tensorflow:	 2.9.1
matplotlib:	 3.5.2
Numpy:		 1.21.5
Keras:           1.1.2
Pillow:		 9.2.0

## Example Input and Output

