# AKOA Knee Laterality Classification with Perceiver Transformer

![img.png](img.png)

Perceiver Transformer Architecture:
-
The Perceiver Transformer is a type of transformer network which relies on cross-attention layers
instead of self-attention layers. This helps reduce the dimensionality of the model such that


The constructed Perceiver model in the current report has the following structure:



Requirements:
- 
Tensorflow version >= 2.4.1
  - i.e. conda install tensorflow-gpu
- tensorflow-addons i.e. pip install -U tensorflow-addons
- tensorflow-datasets i.e. pip install tensorflow-datasets
- scikit-learn i.e. pip install scikit_learn

- conda cudatoolkit 11.0 (need to install by conda if not installed on local gpu)
- cudnn 8

Training:

The images in the AKOA dataset are of size _ * _, which are downsampled to 16 * 16
for the training and validation of the model. 

![input_train_image.png](input_train_image.png)


![test_acc.png](test_acc.png)
![Successful_Training_1.png](Successful_Training_1.png)


References:
-