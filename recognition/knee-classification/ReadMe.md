#  Welcome to the Pattern Recognition Project

## Project Name

Classification laterality on knee data set


## Project Overview

The objective for this project is to classify the laterality (left or right sided knee) of the OAI AKOA knee image data set, with a minimum accuracy of 0.9 on the test set. All the images are already labeled with either "left" or "right". 


## Dependencies

* Python 3.7
* TensorFlow 2.3


## Data Split

There are over 20000 original images available labelled into two categories: left and right. The dataset is split into training set and test set with the ratio of 8:2. For more details about the dataset, please refer to the link: https://nda.nih.gov/oai/

## Algorithm: 

Inception V3 by Google is the 3rd version in a series of Deep Learning Convolutional Architectures. Inception V3 was trained using a dataset of 1,000 classes from the original ImageNet dataset which was trained with over 1 million training images. On top of the Inception 3 base model, I use the principle of transfer learning. The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. In the project, I take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

I followed the general machine learning workflow.

* Examine and understand the data
* Build an input pipeline, in this case using Keras ImageDataGenerator
* Compose the model
* Load in the pretrained base model (and pretrained weights)
* Stack the classification layers on top
* Train the model
* Evaluate model

Here are a list of fine-tunings that are specific to the project:

* Data Augmentation
* Image data pre-processing
* Un-freeze the top layers of the model

## Example outputs
### Testing accuracy
![WechatIMG65](https://user-images.githubusercontent.com/45050125/97295079-26f4ea00-184f-11eb-87f9-f6d23191c250.png)

### Training and testing curve
![WechatIMG66](https://user-images.githubusercontent.com/45050125/97295094-2a887100-184f-11eb-93a3-7baba9f2a01f.png)

