# Visual Transformer (ViT) for classifying Alzheimer's Disease

COMP3710 Report
Eric Shen
46466369


| Alzheimer's | Normal |
| --- | --- |
| ![Alzheimer](./images/AD.jpeg) | ![Normal](./images/NC.jpeg) |
Images of the human brain from the [ADNI](https://adni.loni.usc.edu/) dataset, as classified by a Visual Transformer.

## Project Aims
This project aims to classify images of the brain from the ADNI dataset as either exhibiting Alzheimer's disease or not. This was implemented using a Visual Transformer (ViT) model, based on the architecture defined in [this paper](https://arxiv.org/pdf/2010.11929.pdf "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"), and the inspired by the implementation in [this article](https://keras.io/examples/vision/image_classification_with_vision_transformer/).

## Model Architecture
![Model Architecture](./images/vit_model_architecture.png "ViT model architecture from the above paper")

The model takes images as input, breaking it up into flattened patches which are then fed into a transformer encoder. Each of these patches are given a position embedding to retain information about order. The transformer encoder uses an attention mechanism to retain information about previously viewed data, and is in theory capable of keeping this memory of extremely large sets of data. This allows the model to learn to associate certain patches of an image with others, based on factors such as their positioning. After being run through the transformer encoder, the output is fed to a MLP which classifies the image.



## Requirements
tensorflow 2.10.0

keras 2.10.0

Pillow 9.0.1

tensorflow-addons 0.18.0

zipp 3.8.1

matplotlib 3.5.1

## Installation & Usage


## Papers and references
### Papers
[Vaswani A. et al, Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[Dosovitskiy A. et al, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)

### Code references
[Display Deep Learning Model Training History in Keras](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)

[Vision Transformer in TensorFlow](https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html)

[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)