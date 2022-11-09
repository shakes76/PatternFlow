# GCN for Facebook Large Page-Page Network

 Pytorch implementation of Graph Convolution Network to solve report problem 2. 

This GCN is a naive implementation of paper shown below:

Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

## How graph convolution network works?

GCN, graph convolutional neural network, actually has the same function as CNN, which is a feature extractor, but its object is graph data structure. GCN has cleverly designed a method to extract features from graph data, so that we can use these features to perform node classification on graph data.



<img src="https://raw.githubusercontent.com/Flashfooox/PatternFlow/topic-recognition/recognition/s4589963_GCN/images/IMG_7459.PNG">



The GCN in the figure above input a graph, and the features of each node change from X to Z through several layers of GCN. However, no matter how many layers there are, the relation of connection between nodes, i.e., A, is shared.

The input of each layer of GCN is the adjacency matrix A and node feature H, so we can directly do a product and multiply it by A parameter matrix W, and then apply it to activation function. 
$$
f(H^{(l)}, A)=\sigma(AH^{(l)}W^{(l)})
$$
This is quiet similar to an simple neural network layer.

But this simple model has several limitations:

-  If only use A, since the diagonal of A is 0, when multiplying with the feature matrix H, only the weighted sum of the features of all the neighbours of A are calculated, while the node's own features are ignored. So, we can add the identity matrix I to A, so that the diagonal element becomes 1.

- A is an unnormalized matrix, so multiplying by the eigen-matrix will change the original distribution of features and cause some unpredictable problems. So, we should normalize A. 

Finally, by improving the above two limitations we get our layer-to-layer propagation equation:
$$
H^{(l+1)}=\sigma(\widetilde{D}^{-\frac{1}{2}}\widetilde{A}\widetilde{D}^{-\frac{1}{2}} H^{(l)}W^{(l)})
$$

- A~=A + I, A is adjacency matrix and I is identity matrix.
- D~ is degree matrix of A.
- H are features of each layer.
- W is weight matrix.


Then, the layer-to-layer propagation equation can help us to extract the features.



Let us construct a two-layer GCN, and the activation function adopts ReLU and Softmax respectively, then the forward propagation formula is as follows:
$$
Z=f(X,A)=softmax(\hat{A}Relu(\hat{A}XW^{(0)})W^{(1)})
$$
Finally, we calculate the cross entropy loss function for all labelled nodes:
$$
L = -\sum_{l∈y_{L}}\sum_{f=1}Y_{lf}\ln{Z_{lf}}
$$

Then we can train an node-classification model.

The author call their approach semi-supervised classification because even a small number of nodes can be trained with labels.

## requirements

- pytorch (v1.10.0)
- numpy
- Matplotlib
- scikit-learn

## Data set

Facebook Large Page-Page Network data set consists of 

- 22,470 nodes

- 171,002 edges

- 128 dimension features

- 4 classes

https://snap.stanford.edu/data/facebook-large-page-page-network.html (B. Rozemberczki, C. Allen and R. Sarkar. Multi-scale Attributed Node Embedding. 2019.)

In this problem we used partially processed dataset from report specification.



We split the dataset as showing below:

Training set: 0-3000

Validation set: 3000-4000

Test set: 4000-5000

## How to implement?

- Load dataset:

  - generate the adjacency matrix

  - generate features matrix
  - encode labels to one hot.

- Define GCN layer:
  - implement reset_parameters function to reset parameters.
  - define forward propagation function. (multiply adjacency matrix and support matrix)

- Define GCN model:
  - we use two layers of GCN layer for our model. The input dimension of first layer is 128 (128 dimension features) and output dimension is 16. The input dimension of second layer is 16 and output dimension is 4 (4 classes).
  - define forward propagation function. (apply activation function)

- Train the model:
  - define loss function. (we use cross entropy loss function as shown above)
  - calculate loss value and back-propagate to optimize the loss function. (we use Adam for out optimizer)

## Run the script

Train the model:

```
cd s4589963_GCN
python train.py [dataset path]
```

Train the model and see the visualization:

```
cd s4589963_GCN
python tsne.py [dataset path]
```

## Examples:

By running the training script:

```
======================================================================================
Epoch   1/200: Loss 1.5027, Train_accuracy 0.3907, Validation_accuracy 0.3760, Time 0.1381
Epoch   2/200: Loss 1.4003, Train_accuracy 0.4277, Validation_accuracy 0.4120, Time 0.1421
Epoch   3/200: Loss 1.3221, Train_accuracy 0.4923, Validation_accuracy 0.4810, Time 0.1491
Epoch   4/200: Loss 1.2473, Train_accuracy 0.4823, Validation_accuracy 0.4640, Time 0.1491
Epoch   5/200: Loss 1.2104, Train_accuracy 0.5483, Validation_accuracy 0.5310, Time 0.1511
Epoch   6/200: Loss 1.1617, Train_accuracy 0.5473, Validation_accuracy 0.5480, Time 0.1401
Epoch   7/200: Loss 1.1119, Train_accuracy 0.6023, Validation_accuracy 0.6020, Time 0.1431
Epoch   8/200: Loss 1.0577, Train_accuracy 0.6470, Validation_accuracy 0.6460, Time 0.1461
Epoch   9/200: Loss 1.0095, Train_accuracy 0.6777, Validation_accuracy 0.6760, Time 0.1421
Epoch  10/200: Loss 0.9615, Train_accuracy 0.7223, Validation_accuracy 0.7360, Time 0.1411
Epoch  11/200: Loss 0.9058, Train_accuracy 0.7343, Validation_accuracy 0.7530, Time 0.1431
Epoch  12/200: Loss 0.8571, Train_accuracy 0.7473, Validation_accuracy 0.7690, Time 0.1561
Epoch  13/200: Loss 0.8066, Train_accuracy 0.7613, Validation_accuracy 0.7830, Time 0.1491
Epoch  14/200: Loss 0.7552, Train_accuracy 0.7930, Validation_accuracy 0.8090, Time 0.1391
Epoch  15/200: Loss 0.7067, Train_accuracy 0.8180, Validation_accuracy 0.8270, Time 0.1511
Epoch  16/200: Loss 0.6710, Train_accuracy 0.8247, Validation_accuracy 0.8380, Time 0.1451
Epoch  17/200: Loss 0.6290, Train_accuracy 0.8143, Validation_accuracy 0.8270, Time 0.1321
Epoch  18/200: Loss 0.5956, Train_accuracy 0.8190, Validation_accuracy 0.8320, Time 0.1331
Epoch  19/200: Loss 0.5658, Train_accuracy 0.8377, Validation_accuracy 0.8460, Time 0.1311
Epoch  20/200: Loss 0.5315, Train_accuracy 0.8517, Validation_accuracy 0.8500, Time 0.1481
Epoch  21/200: Loss 0.5085, Train_accuracy 0.8527, Validation_accuracy 0.8550, Time 0.1441
...
```

Use TSNE to do the dimension reduction and observe the classification results:

<img src="https://raw.githubusercontent.com/Flashfooox/PatternFlow/topic-recognition/recognition/s4589963_GCN/images/GCN_tsne.PNG">

We can see that the classification results are basically correct even though we only choose 3,000 for our training set.



Visualize the loss and accuracy curve:

<img src="https://raw.githubusercontent.com/Flashfooox/PatternFlow/topic-recognition/recognition/s4589963_GCN/images/training_effect.png">
