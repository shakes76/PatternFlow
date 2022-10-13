# StyleGAN Implementation
:wave: This StyleGAN implementation is submitted as a response to one of the assessments of COMP3710 of The University of Queensland (UQ), semester 2, 2022 :+1:. 

The task is "generative model of one of the OASIS brain, ADNI brain or the OAI AKOA knee data set using a variant of StyleGAN".

## Development Environment
 - Python version: 3.8.13
 - Tensorflow version: 2.8.0
 - IDE: VSCode 1.71.2

## Training Dataset
All three training image sets were black and white, preprocessed, provided by the lecturer, downloaded from UQ BlackBoard, detailed below：
 - OASIS brain, 11328 images, resolution 256 x 256.
 - ANDI brain (NC), 11120 images, resolution 256 x 240.
 - OAI AKOA knee, 18680 images, resolution 260 x 228.

## Code Structure
This implementation consists of 6 python files,
 - `clayers.py` customized layers, including classes and functions of layers operations, also callbacks.
 - `config.py`  the config file.
 - `dataset.py` includes a class that loads the training dataset.
 - `modules.py` the StyleGAN model implementation, uses components defined in clayers.py.
 - `predict.py` includes functions that load the trained model and generate images.
 - `train.py`   includes training procedures and necessary callback definitions.

## How to train?
### Before training
A few parameters have to be specified in `config.py`.

| Variable            | Description                                                 | Example
| -------------       | -------------                                               |------------- 
| CHANNELS            | Number of channels of training images (grascale or RGB).    |1
| LDIM                | Dimension of latent vectors.                                |128
| SRES                | Starting training resolution, 4 or 8 suggested.             |4
| TRES                | Target training resolution, must be the power of 2.         |256
| BSIZE               | Batch size of each resolution training.                     |(32, 32, 32, 32, 16, 8, 4)
| FILTERS             | Number of filters of each resolution during training.       |(256, 256, 256, 256, 128, 64, 32)
| EPOCHS              | Number of epochs to train for each resolution.              |(25, 25, 25, 25, 30, 35, 40)
| INPUT_IMAGE_FOLDER  | Folder of training images.                                  |D:\images\ADNI_AD_NC_2D
| NSAMPLES            | Number of images to generate for inspection during training.|25
| IMAGE_DIR           | Directory of generated images during training.              |C:\output
| MODEL_DIR           | Directory of plotted models during training.                |C:\output\images
| CKPTS_DIR           | Directory of to save checkpoints.                           |C:\output\models
| LOG_DIR             | Directory to save output loss logs.                         |C:\output\logs

### Start training
Training can be run by simply nevigating to the project folder and executing `python train.py`.

### During training
`NSAMPLES` sample images will be generated after each epoch under the folder `IMAGE_DIR` as configured in `config.py`. 4 model plots, fade-in discrimator, stabilized discriminator, fade-in generator, stabilized generator will be generated in `MODEL_DIR` for each resolution training. Weights will be saved in `CKPTS_DIR` after each resolution training.

### After training
Two csv files, dloss.csv and gloss.csv, of log of training loss will be generated in `LOG_DIR`, from which plots can be generated.
