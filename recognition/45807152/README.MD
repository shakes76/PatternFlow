_45807152 LAUGHER, George_

# Improved UNet Model for Segmenting the ISIC Lesion Data Set
This recognition solution features an algorithm documented as an [Improved UNet Model](https://arxiv.org/pdf/1802.10508v1.pdf) and utilises the [2017 ISIC Challenge Dataset](https://challenge.isic-archive.com/data/#2017) to perform skin lesion segmentation.

## The Problem Scope
The utilisation of deep learning recognition systems in the medical field are becoming increasingly popular and in demand due to their increasing performance in recent years. In particular, with skin cancer being one of the most common types of cancer in Australia, efficient and effective skin cancer screening is vitally important. Thus, using a deep learning (DL) framework and inspiration from the [Improved UNet Model](https://arxiv.org/pdf/1802.10508v1.pdf) research paper, this recognition solution aims to segment images of skin lesions with a minimum [Dice similarity](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) coefficient of **_0.8_** when testing. The dice similarity coefficient in addition to the [Intersection over Union (IoU)](https://en.wikipedia.org/wiki/Jaccard_index) metric will be used to evaluate model performance and compare the expected segmentation mask to the models predicted mask. In both metrics, _1.0_ represents a _100% match_.

## Algorithm Description and Function
The improved UNet model is a convolutional neural network (CNN) that features a 'U' shape that decodes an imput image through various convolution and context layers, then, re-encodes a segmentation mask through upsampling and localisation layers. As shown in the below model architecture, the **localisation pathway** (_right-hand-side_) of the CNN recieves gradient signals from the **context pathway** (_left-hand-side_) through injections at each level for enhanced deep learning.

![Improved UNet Network Architechture](./figures/ImprovedUNetArchitecture.png)

Levels represent the vertical depth within the 'U' shaped with higher levels being of lower spatial resolution, but higher feature representation. In more detail, 3x3x3 convolution layers are used with a stride of 2 to reduce the resolution of the image during the context pathway. Furthermore, during the localisation pathway, upsampled features are recombined with the featrures from the corresponding level of context aggregation via concatenation.

## How It Works
The recognition solution is implemented using TensorFlow _(version 2.9.1)_ and divided into multiple Python files,
* **modules.py** - contains Improved UNet model class and components. From here, the UNet CNN can be built.
* **dataset.py** - provides a data loader for the 2017 ISIC data. Performs preprocessing to image and mask data.
* **train.py** - used to trin the Improved UNet model on the 2017 ISIC data. Provides useful plots, metrics and prediction comparisons to evaluate segmentation performance.
* **predict.py** - contains functions to predict segmentation mask from a trained model. Provides comparison plots to ground truth.
* **utility.py** - contains metric calculations used when training, validating and testing the Improved UNet model.
