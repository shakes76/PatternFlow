# Classifying Alzheimer's Disease Diagnoses Using Vision Trainsformer

This project attempts to classify the [ADNI](https://adni.loni.usc.edu/) brain dataset into AD (Alzheimer's Disease) and NC (Normal Cognitive). It uses the Vision Transformer (ViT) as proposed by the paper [An Image is Worth 16x16 Words](http://arxiv.org/abs/2010.11929). The original paper is implemented with JAX, but I used Tensorflow, so I used the implementation by [Saptashwa Bhattacharyya](https://towardsdatascience.com/understand-and-implement-vision-transformer-with-tensorflow-2-0-f5435769093). The dataset was split into 3 groups: 21000 images for training, 6000 for validation, and 3000 for testing. Results were somewhat dissapointing, as I only achieved a validation accuracy of __%, and a testing accuracy of __%. I even tried multiple different preprocessing techniques, but these only made the accuracy worse.

## Vision Transformer

![Vision Transformer gif](images/vit.gif)

