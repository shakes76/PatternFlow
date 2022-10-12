# StyleGAN Implementation
:wave: This StyleGAN implementation is submitted as a response to one of the assessments of COMP3710 of The University of Queensland (UQ), semester 2, 2022 :+1:. The chosen task is the "generative model of one of the OASIS brain, ADNI brain or the OAI AKOA knee data set using a variant of StyleGAN".

## Development Environment
 - Python version: 3.8.13
 - Tensorflow version: 2.8.0
 - IDE: VSCode 1.71.2

## Training Dataset
All three training images were black and white, preprocessed, provided by the lecturer, downloaded from UQ BlackBoard, detailed below：
 - OASIS brain, 11328 images, resolution 256 x 256.
 - ANDI brain (NC), 11120 images, resolution 256 x 240.
 - OAI AKOA knee, 18680 images, resolution 260 x 228.

## Code Structure
This implementation consists of 6 python files,
 - `clayers.py` includes customized layers, including classes and functions of layers operations.
 - `config.py`  the config file.
 - `dataset.py` includes a class that loads the training dataset.
 - `modules.py` the StyleGAN model implementation, uses components defined in clayers.py.
 - `predict.py` includes functions that load the trained model and generate images.
 - `train.py`   includes training procedures and necessary callback definitions.
