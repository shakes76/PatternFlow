# Stable Diffusion generative model on OASIS brain dataset
---


Diffusion is a new class of generative technique. The idea was proposed by (Sohl-Dickstein, et al.) in 2015, where  a model generates an image from Noise.

This is a Tensorflow implementation of Stabel Diffusion on the OASIS brain dataset.

The main idea in this implimentation is from [Image generation with diffusion models using Keras and TensorFlow](https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac).

The code of this project can be also found in the [Google Colab](https://colab.research.google.com/drive/1ebIE7dlwSnsaJe7G0ngCuGTSa9qdSVKz?usp=sharing) .  
<br>

## Environment Requirements
---
* OS: Windows10/11 64bit, Linux 64bit. (Excluding MacOS as tensorflow_addons has no MacOS version on anaconda currently. Google Colab can be utilized on MacOS for alternative)
* Python 3.7+
* Tensorflow 2.8.1 +  
<br>

## Installation
---
### &nbsp; 1. Install the Tensorflow
If there is no Tensorflow in your computer, please install the Tensorflow 2.8.1+.
```
$ pip install tensorflow
```
If you use Anaconda, use this command:
```
$ conda install -c conda-forge tensorflow
``` 
### &nbsp;2. Install tensorflow_addons
The tesnorflow_addons has no MacOS version in anaconda. Install tensorflow_addons using the following command:
```
$ pip install tensorflow_addons
```
If you use Anaconda, use this command:
```
$ conda install -c esri tensorflow-addons
```
### &nbsp;3. Install einops
The rearrange function in einops library is useful for time embedding in Attention.Install einops using the following command:
```
$ pip install einops
```
or
```
$ conda install -c conda-forge einops
```  
<br>

## Code Usage
---
There are four .py files in this repository:
* dataset.py
* modules.py
* predict.py
* train.py

The dataset.py and modules.py are support files including some support functions. The predict.py and train.py files are excuting files for predicting a picture and training a model from the OASIS brain dataset, respectively.

### &nbsp; Parameter Setting  
#### &nbsp;&nbsp; 1. train.py  
* PATH = "./dataset/keras_png_slices_data.zip"  
    The path of zipped image dataset  
* NUMBER_OF_SAMPLES = 2000  
    The number of image samples used to training  
* IMAGE_SIZE = (128, 128)  
    The size of training image.
* CKPT_PATH = "./checkpoint"
    The path of the checkpoint  
* EPOCS = 30  
    The training epochs.
* BATCH_SIZE = 16  
    The training batch size.  
#### &nbsp;&nbsp; 2. modules.py  
* timesteps = 1000  
    The totla time steps of adding noise / denoising  

The reason why I didn't use argument transmitting from python command is the predict function use some parameters imported from train.py. The predict.py will read the path of the checkpoint set in the train.py file. I will use the argument transmitting in the following version.  

### &nbsp; Code Excution  
After setting the parameters, open the Terminal (Command line) under this file folder, then type the command as follows:  
* training model:
```
$ python train.py
```
* generating a picture:
```
$ python predict.py
```  

<br>

## Stable Diffusion  
---

The modle of Stabel Diffusion is shown as follow:
![](/imgs/stable_diffusion.png)

dd

dd

![](/imgs/ddpm64.gif)