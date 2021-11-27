# Graph Convolutional Networks
This is a PyTorch implementation of GCN model to carry out a semi supervised multi-class 
node classification using Facebook Large Page-Page Network dataset. 

References:
Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)](https://arxiv.org/abs/1609.02907)

## Description of algorithm
GCN is the first-order local approximation of spectral convolution. It is a multi-layer graph convolution neural network. Each convolution layer only processes the first-order neighborhood information. The information transmission of multi-order neighborhood can be realized by stacking several convolution layers. The magic of GCN is that it can aggregate the node features near a node and learn the node features through weighted aggregation, so as to do a series of prediction tasks.
![Alt text](https://github.com/shunyuLiu/pract/blob/master/Screenshot.png)
Reference: [https://paperswithcode.com/method/gcn](https://paperswithcode.com/method/gcn)<br />
Suppose we have a batch of graph data, in which there are n nodes, and each node has its own feature dimensions, we set the features of these nodes to form an n * d dimensional characteristic matrix X, and then the relationship between each node will also form an n × n dimension adjacency matrix A. X and A are the inputs to our model.
For all nodes, H<sup>(l)</sup> represents the eigenvector matrix of all stages in layer l, H<sup>(l+1)</sup> represents all eigenvector matrices after one convolution operation. The formula of one-time convolution operation is as follows:
![Alt text](https://github.com/shunyuLiu/pract/blob/master/Screenshot2.png)<br />
Reference: [https://medium.com/skylab-air/text-classification-using-graph-convolutional-networks-9a3b0479ada1](https://medium.com/skylab-air/text-classification-using-graph-convolutional-networks-9a3b0479ada1)<br />
In fact, it realizes the weighted summation of the neighbor nodes of each node in the graph, and uses the multiplication with the parameter matrix to obtain the characteristics of the node of the new layer. The features of each node are iteratively calculated in the form of matrix, then convolution is carried out through layer propagation, and finally the features of node are updated.


## Data
Using Facebook Large Page-Page Network dataset:
- 22470 nodes
- 171002 edges
- 128 dim vectors
- [Partially processed dataset](https://graphmining.ai/datasets/ptg/facebook.npz) will be used, which contains three npy files.

## Implement
- Load data
  - N * N adjacency matrix (N: number of nodes)
  - N * D features matrix (E: number of features)
  - labels to one hot
- GCN layers
  - similar to the CNN layers
  - reset_parameters function to ensure the inital parameters are the same for every test
  - forward function uses A * X * W
- GCN model
  - two layers of GCN model
    - first layers input dimension: 128; output dimension: 16
    - first layers input dimension: 16; output dimension: 4
  - forward function to getpPropagation mode: Relu -> Fropout -> gc2 -> softmax
- Train
  - loss function only calculate the nodes in train set
  - back propagation to optimize loss function

## Requirements
- PyTorch 0.4 or 0.5
- python 2.7 or 3.6

## Run the model
```
cd s4571084

python gcn.py
```
## Visualization
Rnning the training script:<br />
![Alt text](https://github.com/shunyuLiu/pract/blob/master/shotScreen5.jpeg)<br />
Visualize the loss and accuracy <br />
![Alt text](https://github.com/shunyuLiu/pract/blob/master/Screenshot4.png)<br />
TSNE Result:<br />
![Alt text](https://github.com/shunyuLiu/pract/blob/master/Screenshot3.png)<br />




