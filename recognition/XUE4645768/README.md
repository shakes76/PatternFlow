# Graph Convolutional Networks
*COMP3710 Report

*Student Name: Xue Zhang

*Student ID: 46457684

*TensorFlow implementation of Graph Convolutional Networks based on Facebook Large Page-Page Network dataset for semi-supervised multi-class node classification.


# Requirement 

*Python version 3.6 

*Tensorflow 2.5

*Pytorch installation

*Sklearn, pandas, numpy,scipy and matplotlib libraries 






# Data
Facebook Large Page-Page Network
https://snap.stanford.edu/data/facebook-large-page-page-network.htm
    
Processed dataset where the features are in the form of 128 dim vectors .

Data Structure:

Shape of Edge data: (342004, 2)

Shape of Feature data: (22470, 128)

Shape of Target data (22470,)

Number of features of each node: 128
    
Categories of labels: {0, 1, 2, 3}
    
Data split:
Training set : Validation set : Test set = 0.2 : 0.2 :0.6

# Running  

In the gcn.py:Data preprocessing, accuracy of model training  &model test, TSNE embeddings plot with ground truth in colors.
A main function is included in the code

python gcn.py

Warning: Please pay attention to whether the data path is correct when you run the gcn.py.


```python

```
