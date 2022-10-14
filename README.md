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
| OUT_ROOT            | Root folder that contains training outputs.                 |C:\output

> **Note** `OUT_ROOT` folder must not exist. Training will create `OUT_ROOT` folder and 4 sub-folders in it. They are,
 - **ckpts** for saving checkpoints
 - **images** for saving progressive images
 - **log** for saving los loss files
 - **models** for saving model plots

### Start training
Training can be run by simply nevigating to the project root folder and executing **`python train.py`**.

### During training
`NSAMPLES` sample images will be generated after each epoch under the folder `IMAGE_DIR` as configured in `config.py`. 4 model plots, fade-in discrimator, stabilized discriminator, fade-in generator, stabilized generator will be generated in `MODEL_DIR` for each resolution training. Weights will be saved in `CKPTS_DIR` after each resolution training.

### After training
Two csv files, dloss.csv and gloss.csv, of log of training loss will be generated in `LOG_DIR`, from which plots can be generated.

## The Model
My model is built based on [<ins>Progressive Growing GAN</ins>](https://arxiv.org/abs/1710.10196), where each resolution is trained before a higher resolution block fades in (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L152)). Most of architecture follows the [<ins>StyleGAN</ins>](https://arxiv.org/abs/1812.04948) paper but with small variations.

### Overall Structure
The structure of the model is given below.

<p align="center">
    <img src="asset/StyleGAN_Structure.jpg" width="400">
</p>

A few points to note,
 - Dimension of latent vector z is 512.
 - w is transformed and injected 2 times in each resolution block (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L136)).
 - latent vector z is passed through fully connected layers to generate w (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L30) and [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L196)).
 - Number of fully connected layers is 8, w and z have the same dimension.
 - Input of 'Synthesis network' is constant (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L186)).
 - A noise vector is injected 2 times in each resolution block (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L130)).
 - AdaIN (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/645897586b76a0b96dc23ec2ddb7ac442f33d445/clayers.py#L66)) takes 2 inputs, result of conv3x3 + noise and a style vector (see [<ins>here</ins>](https://github.com/KaiatUQ/StyleGAN/blob/e7d4111eae9fadbe16f9431b2524d6f1093f9627/modules.py#L136)).
 - Starting resolution is 4, target resolution is 1024.

### Model Variations
Original model aims to generate photo realistic images of resolution 1024 x 1024 x 3. The dimension of image in my training datasets is considerably smaller (256 x 256 x 1 appox) so my model is simplified accordingly to avoid unnecessary complication which saves training time.
|                             | My Model                                   | Original Model
| -------------               | -------------                              |------------- 
| Dimension of latent vector  | 128                                        | 512
| Image channel               | 1                                          | 3
| Target resolution           | 256 x 256                                  | 1024 x 1024
| Number of filters           | 256, ..., 32                               | 512, ..., 32
| Number of FC layers         | depth of model (6)                         | 8
| Upsampling Mmethod          | Upsample2D                                 | Bilinear


## Reference
* Progressive Growing of GANs for Improved Quality, Stability, and Variation, 2018. [<ins>https://arxiv.org/abs/1710.10196</ins>](https://arxiv.org/abs/1710.10196)
* A Style-Based Generator Architecture for Generative Adversarial Networks, 2019. [<ins>https://arxiv.org/abs/1812.04948</ins>](https://arxiv.org/abs/1812.04948)
