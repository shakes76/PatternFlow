# Improved UNet for ISICs Data
__**Created by Matthew Costello, 45314838**__

This is my implementation of ['the improved UNet'][https://arxiv.org/pdf/1802.10508v1.pdf], adapted for 2D binary class segmentation. It is a convolutional neural network that outputs an original image as a mask, segmented into white and black. Specifcally, the parameters of this model were tuned for the [ISIC 2018 challenge data for skin cancer][https://challenge2018.isic-archive.com/] - this model is capable of segmenting skin lesions with an **average [Dice similarity coefficient][https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient] of about 0.95** (example outputs visualised below).

Explain how model works

Explain data split sets

## Dependencies
My environment.yml file is provided in this directory for ease of use with Anaconda. The following are required:
1. Python 3.7
2. Tensorflow-gpu 2.1
3. Numpy
4. Matplotlib
5. Tensorflow_addons 0.9.1<sup>1</sup> (note that this isn't included within Conda, but pip can be used to add it to the Conda env - v0.9.1 required for TF 2.1)

## Visualisation
![Figures](resources/visuals.jpg?raw=true "Title")


\[1\] 'tensorflow-addons' is an officially supported repository implementing new functionality. More info at https://www.tensorflow.org/addons. Version 0.9.1 is required for TF 2.1. TFA allows for a InstanceNormalization layer (rather than a BatchNormalization layer), as was implemented in the referenced 'improved UNet'. This layer is necessary due to the usage of my small batch-size of 2, which can lead to "stochasticity induced ...\[which\]... may destabilize batch normalizaton" - F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1. While BatchNormalization normalises across the batch, InstanceNormalization normalises each batch separately.
