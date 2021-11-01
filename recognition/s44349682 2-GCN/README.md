# COMP3710 Report
- Name: Bodhi Howe
- Student Number: 44349682
## Problem 2
Multilayer GCN classification of Facebook Large Page-Page Network dataset

## Problem Description
The Facebook Large Page-Page Network dataset is a connected undirected graph of linked Facebook pages, [link](https://graphmining.ai/datasets/ptg/facebook.npz).

Each page is provided with a set of features representing extracted information from the page's description.

All pages within the dataset belong to one of 4 categories: Politicians, Government Agencies, TV Shows or Companies.

The model developed in this report uses a Graph Convolutional Network (GCN) structure to classify each of the pages into one of these four categories.

### Dataset Specifications
The dataset contains three files:
- Features: A 22,470 x 128 matrix, representing all 128 features of each of the 22470 nodes (pages)
- Edges: A 171,002 x 2 matrix, representing all links between nodes
- Target: A 22,470 long list of integers in (0, 1, 2, 3), representing the category of each node

### Files
- `model.py`: Contains the GCN model and Graph Convolutional Layer classes.
- `drive.py`: Processes the data and trains the GCN model defined in `model.py`, then outputs the metrics.

### Explanation of the model
The GCN works by running the graph through multiple convolutional layers to finally generate a prediction of the class for each node.

Each convolutional layer takes an input containing a matrix of the features for each node and the edges between nodes, then performs matrix operations to predict the class of the node based on the nodes features and the features of the nodes adjacent (connected) to it.

The image below gives an example for this process: The node A is given an updated list of features based on the average messages (aggregated features) of the nodes it is connected to (B, C, D). This is applied in turn to each of these nodes as well, using the original values.
![Convolution](https://perfectial.com/wp-content/uploads/2021/01/GNN-01-scaled.jpg)
https://perfectial.com/blog/graph-neural-networks-and-graph-convolutional-networks/
For each additional convolutional layer, the updated features of each node spread to nodes one connection further away.
Thus, typically GCNs only contain a few layers, to avoid over smoothing the data by spreading the updates to very distant nodes, which would typically be considered independent.

The matrix operation which computes this feature update is shown below:
![Layer Output](https://lh5.googleusercontent.com/fCocp4xdLQkhtbKLAbrnnokRoagd_Q2BRKuMCNdwWaGIN-zOL0Mywefl-GOGf0bVllh-got4D3bpnuRpp4eiWp0Be-LsvwozavnyXT6LguFDQ_8QPYp0IPs1T44jwA0pc2PXcPS9)
https://www.experoinc.com/post/node-classification-by-graph-convolutional-network
- Sigma: The activation function
- D^: The degree matrix of the updated Adjacency matrix (A Diagonal matric where D(i,i) = the number of edges from Node i)
- A^: The updated Adjacency matrix, where A(i,j) = 1 if there is an edge between nodes i and j, with added self-loops (edges from a node to itself)
- H: The feature matrix for each node
- W: The weight matrix which is optimized when learning the model

### Setup
The data is broken into a 0.2/0.2/0.6 train/val/test split for semi-supervised learning.
The GCN contains three convolutional layers with dropout between, reducing the 128 features into 64 then 32 hidden layers, before reducing into the 4 classifiers.
The first two layers use ReLU activation while the final uses softmax.

The model was originally set up as seen below, however testing showed improvements to the accuracy and loss metrics after adding the additional convolution and dropout layers.
![Example model](https://lh5.googleusercontent.com/3B2YByoRwIDHupMT8zI2seOkl4ETPP0DySySOV55aF2R5mPyksmbYgLZuXCrAKWJ5OzmtCqpeqXEb409Mf4NMfM7iJ7zhcEpOY5oirZ--Ap8904oleE-Y03xlq8rhIvs5hNBECYM)
https://www.experoinc.com/post/node-classification-by-graph-convolutional-network



## Results
### Training and Validation
#### Training the Model
![Training](https://raw.githubusercontent.com/Sinquios/PatternFlow/topic-recognition/recognition/s44349682%202-GCN/Training.png)
#### Model Accuracy
![Accuracy](https://raw.githubusercontent.com/Sinquios/PatternFlow/topic-recognition/recognition/s44349682%202-GCN/Accuracy.png)
#### Model Loss
![Loss](https://raw.githubusercontent.com/Sinquios/PatternFlow/topic-recognition/recognition/s44349682%202-GCN/Loss.png)
### Prediction of test set
![Testing](https://raw.githubusercontent.com/Sinquios/PatternFlow/topic-recognition/recognition/s44349682%202-GCN/Testing.png)
### TSNE
![TSNE](https://raw.githubusercontent.com/Sinquios/PatternFlow/topic-recognition/recognition/s44349682%202-GCN/TSNE.png)

## Usage
Run the drive.py file to generate the model and receive all outputs seen below.

```
python drive.py
```

Ensure that the `dataset` value in the code contains the path to the `facebook.npz` file linked earlier.

## Libraries
- Tensorflow
- Keras
- Numpy
- Scipy
- Sklearn
- Matplotlib

## Citations
https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b

Kipf, T., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations (ICLR).
https://github.com/tkipf/keras-gcn

https://keras.io/examples/graph/gnn_citations/

https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch 
