# COMP3710 Pattern Recognition Report
### Van Nhat Huy Nguyen (45717309)

## Purpose
The aim of this project is to create and assess the performance of a binary classifier of the OAI AKOA Knee dataset, using the Perceiver model as described by Andrew Jaegle et. al.

The Perceiver paper can be downloaded [here](https://arxiv.org/abs/2103.03206).

This dataset contains 18680 preprocessed X-Ray images of left and right knees of 101 patients. The created model is supposed to distinguish between left and right laterity of the knee X-ray images. 7760 images are labelled as Left, 10920 as Right.

## Implementation
The implementation is divided into 5 main components:
- `cross_attention.py` Implement the cross attention block of the Perceiver model
- `transformer.py`: Implement the self-attention block of the Perceiver model.
- `fourier_encode.py`: Implement the Fourier encoding method that augment the image data with Fourier encoding of the image pixel position.
- `model.py`: Implement the Perceiver model. It consists of a Fourier encoding layer, cross attention module, a self-attention (transformer) module, and a binary classification head.
- `driver.py`: Implement data preprocessing, creating the model and carrying out the training, testing and plotting process.

### Cross Attention module
This module is implemented according to the specification listed in the paper, 

### Transformer module
This module is also implemented according to the specification listed in the paper, 

### Fourier encode module


