# Xiaomeng Cai 45490826

## Description
This project is to classify laterality (left or right sided knee) of the OAI AKOA knee data set, and the accuracy on the test dataset is 90.04%. Algorithm is Convolutional Neural Network.

There are three parts to achieve this project.  
The first part ([split dataset](https://github.com/1665446266/PatternFlow/tree/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/split%20dataset)) is data processing. It classifies the raw dataset as two classe(left knee and right knee).   
The second part ([bulid model](https://github.com/1665446266/PatternFlow/tree/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/bulid%20model)) bulids the training model and finally saves the model's training parameters (h5 file).  
The last part ([pridction](https://github.com/1665446266/PatternFlow/tree/topic-recognition/recognition/project(%20OAI%20AKOA%20knee%20dataset)/pridction))loads h5 file (trained_model.h5) to predict model accuracy in test dataset.

The _REAMDE.md_ in these three folders describes more details.
