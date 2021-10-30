# ISICs dataset segmentation with an Improved UNet Model

##### Author: Tompnyx
#### Student Number: 45872093

## Introduction
In this project an improved UNet model is used for image segmentation to identify Melanoma.

## Algorithm Description
The UNet is an architectural model that consists of two paths: Contraction and Expansion. Contraction is the process
of capturing the context of an image by reducing its features. This is done traditionally through stacks of
convolutional and max pooling layers [3]. Expansion is the process of reintroducing information into higher resolution
layers [3]. In the improved UNet model used, other techniques like dropout, batch normalization, leaky ReLU layers, and
segmentation are used to improve upon the initial model.

## Dataset Description
The ISIC 2018 dataset is a dataset that was released as a resource for an automated image analysis tool development
competition [2]. The dataset consists of images split into two different folders: Pictures of skin lesions and masks
that segment the lesions. The goal of this dataset was to promote automated Melanoma detection [2].

## Choices made for the model
I chose to stick closely to the UNet picture specified in [1]. This involved five layers of contraction and five layers
of expansion, with leaky ReLU layers and Batch Normalisation sprinkled throughout. Each contraction layer included a
context module, with the expansion layers containing a localisation module. Finally, segmentation layers taken from
stages in between the up-sampling and localisation modules were summed together (element-wise) at the end of the model.


## How to build and run
To run the program, the downloaded dataset or a hyperlink to the dataset is needed. To allow the program to detect this
dataset, the link saved in the ISIC2018_data_link variable in driver.py should be replaced with the correct directory.
The following dependencies will also be needed
### Dependencies
- Python (Tested: 3.9.7)
- Tensorflow (Tested: 2.7.0)
- matplotlib (Tested: 3.4.3)
- graphviz (Tested: 2.38)
- pydot (Tested: 1.4.2)

## References
[1] "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge", Fabian Isensee,
Philipp Kickingereder, Wolfgang Wick, Martin Bendszus, Klaus H. Maier-Hein, 2018.
[Online]. Available: https://arxiv.org/abs/1802.10508v1. [Accessed: 28-Oct-2021].
[2] "ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection", ISIC, 2018.
[Online]. Available: https://challenge2018.isic-archive.com/. [Accessed: 30- Oct- 2021].
[3] H. Lamba, "Understanding Semantic Segmentation with UNET", Medium, 2019.
[Online]. Available: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47.
[Accessed: 30- Oct- 2021].

