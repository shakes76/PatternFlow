# StyleGAN Implementation
:wave: This StyleGAN implementation is submitted as a response to one of the assessments of COMP3710 of The University of Queensland (UQ), semester 2, 2022. 

The task is "generative model of one of the OASIS brain, ADNI brain or the OAI AKOA knee data set using a variant of StyleGAN".

## Results

### Generated images of the three datasets
<p align="center">
    <kbd><img src="asset/samples.png"></dbd>
</p>

### Bilinear interpolation
<p align="center">
    <kbd><img src="asset/bilinear_interpolation.png" width="800"></dbd>
</p>

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

## How to train your own images?
### Before training
A few parameters have to be specified in `config.py`.

| Variable            | Description                                                 | Example
| -------------       | -------------                                               |------------- 
| CHANNELS            | Number of channels of training images.                      | 1
| LDIM                | Dimension of latent vectors.                                | 128
| SRES                | Starting resolution, 4 or 8 suggested.                      | 4
| TRES                | Target resolution, must be the power of 2.                  | 256
| BSIZE               | Batch size of each resolution training.                     | (32, 32, 32, 32, 16, 8, 4)
| FILTERS             | Number of filters of each resolution.       | (256, 256, 256, 256, 128, 64, 32)
| STAB                | Whether to stabilize-train the model after fade-in.         | False
| EPOCHS              | Number of epochs to train for each resolution.              | {0:50, 1:(40,10), 2:(40,10), 3:(40,10), 4:(40,20), 5:(40,20), 6:(40,20)}
| INPUT_IMAGE_FOLDER  | Folder of training images.                                  | D:\ADNI_AD_NC_2D
| NSAMPLES            | Number of images to generate when training.                 | 25
| OUT_ROOT            | Root folder that contains training outputs.                 | D:\output

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
    <kbd><img src="asset/StyleGAN_Structure.jpg" width="450"></dbd>
</p>

A few points to note,

 - latent vector z is passed through fully connected layers to generate w (see [<ins>here</ins>](modules.py#L184) and [<ins>here</ins>](modules.py#L213)).
 - w is transformed and injected 2 times in each resolution block (see [<ins>here</ins>](modules.py#L127) and [<ins>here</ins>](modules.py#L133)).
 - number of fully connected layers is 8, w and z have the same dimension.
 - Input of 'Synthesis network' is constant (see [<ins>here</ins>](modules.py#L177)).
 - a noise vector is injected 2 times in each resolution block (see [<ins>here</ins>](modules.py#L125) and see [<ins>here</ins>](modules.py#L131)).
 - AdaIN takes 2 inputs, result of conv3x3 + noise and a style vector (see [<ins>here</ins>](modules.py#L125) and see [<ins>here</ins>](modules.py#L131)).
 - used loss function [<ins>Wasserstein Distance</ins>](https://arxiv.org/abs/1701.07875) for gradient stability (see [<ins>here</ins>](modules.py#L196)).
 - Model is trained progressively.

### Model Variations
Original paper aims to generate photo realistic images of resolution 1024 x 1024. The dimension of image in my training datasets is much smaller (256 x 256 appox.) and is in grayscale so my model is a simplified version (in terms of training configurations, not model structure) of StyleGAN, to avoid unnecessary complication, which saves training time.

|                             | My Model           | Original       | Justification
| -------------               | -------------      |-------------   |------------- 
| Dimension of latent vector  | 200                | 512            | Original model trains 1024x1024 images, mine 256x256, reduced for simplification.
| Image channel               | 1                  | 3              | The training images are in grayscale.
| Target resolution           | 256 x 256          | 1024 x 1024    | The training images are in 256x256 or similar scale.
| Number of filters           | 256, ..., 32       | 512, ..., 32   | Reduced unnecessary complexity.
| Number of FC layers         | depth of model (6) | 8              | Reduced unnecessary complexity.
| Upsampling method           | Upsample2D         | Bilinear       | Reduced unnecessary complexity.

## A Training Example

Below is a training trail in my experiment. Out of the three datasets, the OASIS is the easiest to train, but the model is also the easiest to collapse. While AKOA is relatively difficult to train since the trianing images are quite noisy.

### Settings
 - Starting resolution: 4x4.
 - Target resolution: 256x256.
 - Latent vector dimemsion: 200.
 - Batch size: 16, 16, 8, 8, 8, 4, 4.
 - Epochs: 10, 15, 15, 15, 15, 15, 15.
 - Training images: 4800 selected.
 - Model not stabilized after fade in.

### Model Evolution
<p align="center">
    <kbd><img src="asset/training_process.gif" width="550"></kbd>
</p>


### Loss plot
Both discriminator and generator converged well in the lower dimensions, but fluctuated at higher dimensions. Most significant changes in loss were observed when model grew, as highlighted in below plot.

<p align="center">
    <kbd><img src="asset/loss_plot.png" width="550">
</p>

## How to play with `predict.py`?
Once trained, `predict.py` can be used to load trained models and generate above images by running **`python predict.py`**. It supports two types of image generations, nxn random images and bilinear interpolation as shown above. Make sure below parameters are properly configured. 

When then program finishes running, the generated images can be found in the specified folder.

> **Note** make sure the same parameters are used as how model is trained, especially the target resolution has to match. Keep all `.py` files under the same folder, as other files are dependencies of `predict.py`.

```
# PARAMETERS TO SET
output_res = (256, 256)          # output resolution of generated images
n = 9                            # number of samples to generate, squre of an integer
steps = 10                       # steps of interpolation
ckpt = r'path of ckpts'          # path of checkpoint files, ex: r'D:\ckpts\OASIS.ckpt'
folder = r'path to save images'  # path of folder the generated images to be saved
```

:boom: **Trained models that generated above images can be found [<ins>here</ins>](https://drive.google.com/file/d/15CD1oabf8B_Zu1d4a4ixl2ns-zfg1OIv/view?usp=sharing).**

## Reference
* A Style-Based GANs, 2019. [<ins>https://arxiv.org/abs/1812.04948</ins>](https://arxiv.org/abs/1812.04948)
* Progressive Growing of GANs, 2018. [<ins>https://arxiv.org/abs/1710.10196</ins>](https://arxiv.org/abs/1710.10196)
* Wasserstein GAN, 2017. [<ins>https://arxiv.org/abs/1701.07875</ins>](https://arxiv.org/abs/1701.07875)
