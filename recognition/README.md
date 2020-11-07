# Introduction

This is a project realted to do a segment the OASIS brain data set with an Improved UNet with all labels having a minimum Dice
similarity coefficient of 0.9 on the test set.
Under the improved_unet file it will contain two python scripy, which call Script.py and model.py

# Requirement
tensorflow = 2.3.0
python = 3.7.7

Also need to create a file under C drive call content in order to load and save the dataset

<img src="./improved_unet/image/Figure1.png" alt="Data" width="200"/>

## Algorithm
In this project, I have build a improved_unet base on this paper:https://arxiv.org/abs/1802.10508v1
Here are the model summary

<img src="./improved_unet/image/model1.JPG" alt="model1" width="200"/>
<img src="./improved_unet/image/model2.JPG" alt="model2" width="200"/>
<img src="./improved_unet/image/model3.JPG" alt="model3" width="200"/>
<img src="./improved_unet/image/model4.JPG" alt="model4" width="200"/>
<img src="./improved_unet/image/model5.JPG" alt="model5" width="200"/>

And the newwork architecture form the paper

<img src="./improved_unet/image/overall.JPG" alt="All" width="200"/>
