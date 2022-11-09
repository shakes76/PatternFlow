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

# Training

Learning rate= 0.01
Weight dacay =0.005

For 200 epoches:
```Epoch 000: Loss 0.2894, TrainAcc 0.9126, ValAcc 0.8954
Epoch 001: Loss 0.2880, TrainAcc 0.9126, ValAcc 0.895
Epoch 002: Loss 0.2866, TrainAcc 0.9126, ValAcc 0.8961
Epoch 003: Loss 0.2853, TrainAcc 0.9132, ValAcc 0.8961
Epoch 004: Loss 0.2839, TrainAcc 0.9137, ValAcc 0.8961
Epoch 005: Loss 0.2826, TrainAcc 0.9141, ValAcc 0.8963
Epoch 006: Loss 0.2813, TrainAcc 0.9146, ValAcc 0.8956
Epoch 007: Loss 0.2800, TrainAcc 0.9146, ValAcc 0.8956
Epoch 008: Loss 0.2788, TrainAcc 0.9146, ValAcc 0.8959
Epoch 009: Loss 0.2775, TrainAcc 0.9146, ValAcc 0.8970
Epoch 010: Loss 0.2763, TrainAcc 0.915, ValAcc 0.8974
Epoch 011: Loss 0.2751, TrainAcc 0.915, ValAcc 0.8972
Epoch 012: Loss 0.2739, TrainAcc 0.915, ValAcc 0.8976
Epoch 013: Loss 0.2727, TrainAcc 0.9157, ValAcc 0.8979
Epoch 014: Loss 0.2716, TrainAcc 0.9157, ValAcc 0.8983
Epoch 015: Loss 0.2704, TrainAcc 0.9161, ValAcc 0.8990
Epoch 016: Loss 0.2693, TrainAcc 0.9168, ValAcc 0.8988
Epoch 017: Loss 0.2682, TrainAcc 0.9181, ValAcc 0.8990
Epoch 018: Loss 0.2671, TrainAcc 0.9179, ValAcc 0.8990
Epoch 019: Loss 0.2660, TrainAcc 0.9179, ValAcc 0.8992
Epoch 020: Loss 0.2650, TrainAcc 0.9188, ValAcc 0.8996
......
Epoch 190: Loss 0.1623, TrainAcc 0.9553, ValAcc 0.9134
Epoch 191: Loss 0.1619, TrainAcc 0.9555, ValAcc 0.9134
Epoch 192: Loss 0.1615, TrainAcc 0.9555, ValAcc 0.9132
Epoch 193: Loss 0.1611, TrainAcc 0.9557, ValAcc 0.9130
Epoch 194: Loss 0.1607, TrainAcc 0.9562, ValAcc 0.9130
Epoch 195: Loss 0.1603, TrainAcc 0.9559, ValAcc 0.9130
Epoch 196: Loss 0.1599, TrainAcc 0.9562, ValAcc 0.9126
Epoch 197: Loss 0.1595, TrainAcc 0.9562, ValAcc 0.9123
Epoch 198: Loss 0.1591, TrainAcc 0.9562, ValAcc 0.9123
Epoch 199: Loss 0.1587, TrainAcc 0.9562, ValAcc 0.9123```

For test accuracy:around 0.9 

# TSNE
For the test:iteration=500, with lower dimension to 2

<img src="https://github.com/eliasxue/3710-pattern-flow/blob/main/tsne%20.png?raw=true">


```python

```
