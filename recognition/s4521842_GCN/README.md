# Graph Convolutional Networks

 - Student Name: Zhu Gao
 - Student ID: 45218422
 - TensorFlow implementation of Graph Convolutional Networks based on Facebook Large Page-Page Network dataset for semi-supervised multi-class node classification.

## Introduction

GCN (Graph Convolutional Network) is similar with CNN(convolutional neural network) that can work directly on the graph and utilize its structural information. Early variants of neural networks can only be implemented using conventional or Euclidean data, but a large amount of real-world data has a non-Euclidean underlying graph structure, which causes the developments in GCN.

<p align="center">
 <img src="https://github.com/SteveInUQ/PatternFlow/blob/topic-recognition/recognition/s4521842_GCN/GCN/image/GCN_process.png?raw=true">
</p>
 
<br/>

Formally, GCN is a neural network that operates on graphs. Given a graph G = (V, E), GCN takes as input:

- The feature description xi of each node i; summarized in an N × D feature matrix X ( N: number of nodes, D: number of input features)
- A representative description of the graph structure in matrix form; usually in the form of the adjacency matrix A

And then generate output Z ( N × F feature matrix, (F: the number of output features for each node)). 

<br/>

Each neural network layer take input with the adjacency matrix A and feature matrix H, so the simple forward propagation equation is:

<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?H^{(l&plus;1)}=\sigma(AH^{(l)}W^{(l)})"/>
</p>

- W^(l) is a weight matrix for the l-th neural network layer 
- σ is a activation function

<br/>

The simple model has two limitations:

1. When we multiply A, for each note, we add all the feature vectors of all adjacent nodes, not the node itself. Therefore, the identity matrix will be added to A.
2. A is not unnormalized, so if multipling with A, the scale of the feature vectors will be change. Therefore, A should be normalized.

After applying these two solution, we will get a new forward propagation equation:

<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?H^{(l&plus;1)}=\sigma(\widetilde{D}^{-\frac{1}{2}}\widetilde{A}\widetilde{D}^{-\frac{1}{2}}&space;H^{(l)}W^{(l)})"/>
</p>

- 𝐴̂ = A + I
- I is the identity matrix
- 𝐷̂ is degree matrix of 𝐴̂

<br/>

With 3-Layer GCN,the form of the forward model is:

<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?Z&space;=&space;f(X,A)&space;=&space;softmax(\hat{A}ReLU(\hat{A}&space;ReLU(\hat{A}&space;X&space;W^{(0)})&space;W^{(1)})&space;W^{(2)})"/>
</p>

After one-hot encoding, we get a 4-dimensional label dataset, and F will be set to 4. After obtaining 4-dimensional vectors in the third layer, we use the softmax function to predict these vectors.

<br/>

Finally, we use categorical cross-entropy to calculate the error.

<p align="center">
 <img src="https://latex.codecogs.com/gif.latex?\mathrm{Loss}&space;=&space;-\sum_{l&space;\in&space;y_L}&space;\sum_{f=1}^{F}&space;Y_{lf}\ln{Z_{lf}} "/>
</p>

- y(L) is the set of node indices that have labels



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
    - Training set : Validation set : Test set = 0.3 : 0.2 :0.5

## Implementation 

1. Load data:

- load the dataset of Facebook Large Page-Page Network
- normalize the feature data
- build and normalize the adjacency matrix
- convert labels to one-hot encoding
- create Boolean masks for training, validation, and testing dataset. The elements of those masks are True when they belong to corresponding dataset. 

2. build GCN model: 

- layer_1: GraphConvolutionLayer(input_dim=128, output_dim=64, activation=relu)
- layer_2: Dropout(0.3)
- layer_3: GraphConvolutionLayer(input_dim=64, output_dim=16, activation=relu)
- layer_4: Dropout(0.3)
- layer_5: GraphConvolutionLayer(input_dim=16, output_dim=4, activation=softmax)

3. train GCN model:

- GCN model take 2 input, the Node Features Matrix (X) and Adjacency Matrix (A), respectively
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

Epoch_1: loss=1.39647 tr_acc=0.43184 val_acc=0.43836
Epoch_2: loss=1.36725 tr_acc=0.43688 val_acc=0.44326
Epoch_3: loss=1.34257 tr_acc=0.43243 val_acc=0.45127
Epoch_4: loss=1.31326 tr_acc=0.46328 val_acc=0.48331
Epoch_5: loss=1.27634 tr_acc=0.50067 val_acc=0.52292
Epoch_6: loss=1.23863 tr_acc=0.55022 val_acc=0.56609
Epoch_7: loss=1.19732 tr_acc=0.60050 val_acc=0.61282
Epoch_8: loss=1.14876 tr_acc=0.63685 val_acc=0.65087
Epoch_9: loss=1.09742 tr_acc=0.67423 val_acc=0.68447
Epoch_10: loss=1.04080 tr_acc=0.70776 val_acc=0.72385
...
Epoch_91: loss=0.20986 tr_acc=0.94081 val_acc=0.92479
Epoch_92: loss=0.20879 tr_acc=0.93977 val_acc=0.92457
Epoch_93: loss=0.20582 tr_acc=0.94096 val_acc=0.92412
Epoch_94: loss=0.20737 tr_acc=0.94200 val_acc=0.92546
Epoch_95: loss=0.20878 tr_acc=0.94289 val_acc=0.92679
Epoch_96: loss=0.20830 tr_acc=0.94274 val_acc=0.92768
Epoch_97: loss=0.20568 tr_acc=0.94229 val_acc=0.92701
Epoch_98: loss=0.20955 tr_acc=0.94229 val_acc=0.92768
Epoch_99: loss=0.20333 tr_acc=0.94333 val_acc=0.92657
Epoch_100: loss=0.20432 tr_acc=0.94407 val_acc=0.92635
Test Accuracy: tf.Tensor(0.9270138, shape=(), dtype=float32)
```
_________________

## Visualisation

- T-SNE (based on test set): 
  <img src="https://github.com/SteveInUQ/PatternFlow/blob/topic-recognition/recognition/s4521842_GCN/GCN/image/t-sne.png?raw=true">
- Accuracy (Training Loss, Training Accuracy, Validation Accuracy): 
  <img src="https://github.com/SteveInUQ/PatternFlow/blob/topic-recognition/recognition/s4521842_GCN/GCN/image/GCN_history.png?raw=true">

## Reference

[1] T. N. Kipf and M. Welling, [“Semi-Supervised Classification with Graph Convolutional Networks,”](http://arxiv.org/abs/1609.02907) 2016.
