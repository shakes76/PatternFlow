# Graph Convolutional Networks


 - COMP3710 Report-task 2
 - Student Name: Zhu Gao
 - Student ID: 45218422
 - TensorFlow implementation of Graph Convolutional Networks based on Facebook Large Page-Page Network dataset for semi-supervised multi-class node classification.


## Introduction

GCN (Graph Convolutional Network) is similar with CNN(convolutional neural network) that can work directly on the graph and utilize its structural information. 



## Requirements

- TensorFlow (2.5.0)
- Numpy
- Matplotlib
- Scikit-learn

## Run the project

> <code>cd GCN/</code>
> 
> <code>python test_drive.py</code>
> <p>Warning: Please pay attention to whether the data path is correct when you run the test_drive.py.</p>

## Dataset
- Data resource: [Facebook Large Page-Page Network](https://snap.stanford.edu/data/facebook-large-page-page-network.html)
- Processed dataset where the features are in the form of 128 dim vectors ([here](https://graphmining.ai/datasets/ptg/facebook.npz)).
- Data Structure:
    - Shape of Edge data: (342004, 2)
    - Shape of Feature data: (22470, 128)
    - Shape of Target data (22470,)
    - Number of nodes:  22470
    - Number of features of each node: 128
    - Categories of labels: {0, 1, 2, 3}
- Data split: 
    - Training set : Validation set : Test set = 0.5 : 0.25 :0.25

## Implementation 

1. Load data:

- load the dataset of Facebook Large Page-Page Network
- normalize the feature data
- build and normalize the adjacency matrix
- one-hot encode labels  

2. build GCN model: 

- layer_1: GraphConvolutionLayer(input_dim=128, output_dim=64, activation=relu)
- layer_2: Dropout(0.3)
- layer_3: GraphConvolutionLayer(input_dim=64, output_dim=16, activation=relu)
- layer_4: Dropout(0.3)
- layer_5: GraphConvolutionLayer(input_dim=16, output_dim=4, activation=softmax)

3. Train GCN model:

- calculate loss by Categorical Cross entropy function
- calculate gradients
- update weights (Adam: learning rate = 0.01, decay = 5e-5)

4. Accuracy:

- The accuracy of the GCN model achieved 0.9304023.
    
5. Plot result

- Plot training loss, training accuracy and validation accuracy
- Plot t-sne (reduce the test data to 2 dimension by PCA)

## Training process

_________________
```
Shape of Edge data (342004, 2)
Shape of Feature data (22470, 128)
Shape of Target data (22470,)
--------------------------------------------------
Number of nodes:  22470
Number of features of each node:  128
Categories of labels:  {0, 1, 2, 3}

Epoch_1: loss=1.38999 tr_acc=0.56440 val_acc=0.56703
Epoch_2: loss=1.35644 tr_acc=0.66008 val_acc=0.65302
Epoch_3: loss=1.32523 tr_acc=0.68376 val_acc=0.67545
Epoch_4: loss=1.29111 tr_acc=0.69025 val_acc=0.68649
Epoch_5: loss=1.25130 tr_acc=0.69337 val_acc=0.68613
Epoch_6: loss=1.21046 tr_acc=0.69150 val_acc=0.68613
Epoch_7: loss=1.16554 tr_acc=0.69390 val_acc=0.68916
Epoch_8: loss=1.12079 tr_acc=0.70280 val_acc=0.69753
Epoch_9: loss=1.06966 tr_acc=0.71633 val_acc=0.70981
Epoch_10: loss=1.02601 tr_acc=0.72603 val_acc=0.72085
...
Epoch_91: loss=0.24722 tr_acc=0.93574 val_acc=0.92594
Epoch_92: loss=0.24228 tr_acc=0.93556 val_acc=0.92594
Epoch_93: loss=0.24013 tr_acc=0.93485 val_acc=0.92523
Epoch_94: loss=0.24372 tr_acc=0.93627 val_acc=0.92612
Epoch_95: loss=0.24348 tr_acc=0.93778 val_acc=0.92772
Epoch_96: loss=0.24097 tr_acc=0.93823 val_acc=0.92790
Epoch_97: loss=0.23825 tr_acc=0.93769 val_acc=0.92772
Epoch_98: loss=0.24018 tr_acc=0.93734 val_acc=0.92612
Epoch_99: loss=0.23857 tr_acc=0.93752 val_acc=0.92594
Epoch_100: loss=0.23755 tr_acc=0.93885 val_acc=0.92701
Test Accuracy: tf.Tensor(0.9304023, shape=(), dtype=float32)
```
_________________

## Visualisation

- T-SNE (based on test set):
<img src="https://github.com/SteveInUQ/PatternFlow/blob/topic-recognition/recognition/s4521842_GCN/GCN/image/t-sne.png?raw=true">
- Accuracy (Training Loss, Training Accuracy, Validation Accuracy):
<img src="https://github.com/SteveInUQ/PatternFlow/blob/topic-recognition/recognition/s4521842_GCN/GCN/image/GCN_history.png?raw=true">

## Reference

[1] T. N. Kipf and M. Welling, [“Semi-Supervised Classification with Graph Convolutional Networks,”](http://arxiv.org/abs/1609.02907) 2016.


```python

```
