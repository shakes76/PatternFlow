# Stable Diffusion generative model on OASIS brain dataset
---


Diffusion is a new class of generative technique. The idea was proposed by (Sohl-Dickstein, et al.) in 2015, where  a model generates an image from Noise.

Stable Diffusion is an AI-generated text-to-image model that will empower billions of people to create stunning art within seconds. 

This is a Tensorflow implementation of Stabel Diffusion on the OASIS brain dataset.

The main idea in this implimentation is from [Image generation with diffusion models using Keras and TensorFlow](https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac).

The code of this project can be also find in the [Google Colab](https://colab.research.google.com/drive/1ebIE7dlwSnsaJe7G0ngCuGTSa9qdSVKz?usp=sharing) platform

## Environment Requirements
---
* OS: Windows10/11 64bit, Linux 64bit. (Excluding MacOS as tensorflow_addons has no MacOS version currently. Google Colab can be utilized on MacOS for alternative)
* Python 3.7+
* Tensorflow 2.8.1 +

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
pip install tensorflow_addons
```
If you use Anaconda, use this command:
```
conda install -c esri tensorflow-addons
```
### &nbsp;3. Install einops
The rearrange function in einops library is useful for time embedding in Attention.Install einops using the following command:
```
pip install einops
```
or
```
conda install -c conda-forge einops
```
## Code usage
---




## Stable Diffusion

The modle of Stabel Diffusion is shown as follow:
![](/imgs/stable_diffusion.png)

dd

dd

![](/imgs/ddpm64.gif)