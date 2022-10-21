45807152 LAUGHER, George

# Improved UNet Model for Segmenting the ISIC Lesion Data Set
This recognition solution features an algorithm documented as an [Improved UNet Model](https://arxiv.org/pdf/1802.10508v1.pdf) and utilises the [2017 ISIC Challenge Dataset](https://challenge.isic-archive.com/data/#2017) to perform skin lesion segmentation.

## The Problem Scope
The utilisation of deep learning recognition systems in the medical field are becoming increasingly popular and in demand due to their increasing performance in recent years. In particular, with skin cancer being one of the most common types of cancer in Australia, efficient and effective skin cancer screening is vitally important. Thus, using a deep learning (DL) framework and inspiration from the [Improved UNet Model](https://arxiv.org/pdf/1802.10508v1.pdf) research paper, this recognition solution aims to segment images of skin lesions with a minimum [Dice similarity](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) coefficient of 0.8 when testing. The dice similarity coefficient in addition to the [Intersection over Union (IoU)](https://en.wikipedia.org/wiki/Jaccard_index) metric will be used to evaluate model performance and compare the expected segmentation mask to the models predicted mask. In both metrics, 1.0 represents a 100% match.

## Algorithm Description and Function
The improved UNet model is a convolutional neural network (CNN) that features a 'U' shape that decodes an imput image through various convolution and context layers, then, re-encodes a segmentation mask through upsampling and localisation layers. As shown in the below model diagram, the localisation pathway (right-hand-side) of the CNN recieves gradient signals from the context pathway (left-hand-side) through injections at each level.

![Improved UNet Network Architechture](./figures/ImprovedUNetArctitecture.png)
<img src="./figures/ImprovedUNetArctitecture.png" width="400"/> 

## 
