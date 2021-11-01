## Author: Eva Holden, 45797851


##### Pattern Recognition Report 
Task 2 : Create a suitable multi-layer GCN model to carry out a semi supervised multi-class node classification using Facebook Large Page-Page Network dataset [2] with reasonable accuracy. (You may use this partially
processed dataset where the features are in the form of 128 dim vectors.) You should also include a TSNE embeddings plot with ground truth in colors. [Normal Difficulty]



* [Algorithm Description](#Algorithm-Description)<br>
* [Problem Description](#Problem-Description)<br>
* [Libraries Used](#Libraries-Used)<br>
* [Executing the code](#Executing-the-code)<br>
* [GCN Implementation](#GCN-Implementation)<br>
* [GCN Evaluation](#GCN-Evaluation)<br>
* [References](#References)

### Algorithm Description 

Graph Convolutional Networks, known as GCN, use machine learning to extract information from data represented as graphs. A number of tasks can be carried out using graphs, including Graph Classification, Graph Generation, Community Detection, Node Classification and more. Node Classification, which was the focus for this project, uses the information about some of the nodes on the graph and the relationships between these nodes to predict the class of the remaining nodes. This is done by generalising Convolutional Neural Networks to the case of data stored as a graph.

#### Convolutional Neural Networks (CNN)
Convolutional Neural Networks are a specialised type of neural network that use convolution in at least one of their layers instead of matrix multiplication. They generally consist of an input layer, a number of hidden layers and the output layer. The effectiveness of these CNNs is the use of a sequence of filters to extract complex patters, where the patterns get more complex as more filters are used.

#### Graph Convolution
Since transformations of graphs is different to regular transformations on the Euclidean plane, we have to employ elements of graph theory to modify our original Convolution theorem, seen below. 

![Convolution Theorem](https://miro.medium.com/max/900/1*d63IS1Rn8TgYLN2V6XKndw.png)

We define a graph Fourier transform by using the Laplacia matrix as defined in graph theory, which is L=D-A, where D is the degree matrix, a diagonal matrix which contains the number of edges attached to each node, and A is the the adjacency matrix, which is a square matrix indicating for each pair of nodes whether they are connected by an edge. 

![Graph Fourier Transform](https://miro.medium.com/max/1050/1*pu68EGXQJvJRWfwbCFY3vA.png)

So, the graph Fourier transform is the projection of the Laplacian onto the basis defined above.


### Problem Description
#### Dataset : Facebook Large Page-Page Network dataset
The dataset used is [Facebook Large Page-Page Network dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html) which includes information to form a page-page graph of varified Facebook pages, where the nodes represent the pages and the edges are the mutual likes between the pages. In addition to this, there are page features, which have been extracted from the page descriptions. Pages are defined by 4 categories, these being: politicians, government organisations, television shows and companies. The data was collected through the Facebook Graph API in 2017. 

The aim is to train a GCN model on the Facebook Large Page-Page Network Dataset and use this to classify nodes into one of the four categories with reasonable accuracy.


#### Libraries Used
The main libraries used throughout this project are:

[NumPy](https://numpy.org/) is used to load the preprocessed Facebook dataset as it is a .npz file. NumPy was used with the permission of the course tutors.

[Pandas](https://pandas.pydata.org/) is a data analysis and manipulation tool that was used to convert the NumPy arrays to DataSet objects for use in TensorFlow.

The algorithm is built using [TensorFlow](https://www.tensorflow.org/), an open source platform for machine learning. 

[Keras](https://keras.io/) is a neural network library that is used on the top of TensorFlow.

[matplotlib](https://matplotlib.org/) is used primarily for visualisations.

[NetworkX](https://networkx.org/) is a library used for studying graphs, and was used to visualise the graph.

These libraries are used with the permission of the Practical tutors.

#### Executing the code

The src consists of FacebookNetworkGCNN.ipynb, which consists of all of the functions and is the main script to run the algorithm.

### GCN Implementation

#### Facebook Large Page-Page Network dataset
Since we used the partially processed dataset version, data preprocessing consisted only of converting the data to Pandas datasets. The graph can be seen below.

![Graph](https://github.com/EvaHolden/PatternFlow/blob/topic-recognition/recognition/FacebookNetworkGCN/graph.png?raw=true)

#### GCN

The GCN is built using the Keras Functional API. We use Input layers, Dropout layers, and graph Convolution layers. The graph convolution layers are GATConv layers, from the Spektral package. The layers and parameters details are below. 

![GCN Layers](https://github.com/EvaHolden/PatternFlow/blob/topic-recognition/recognition/FacebookNetworkGCN/gcnlayer.JPG?raw=true)

#### Training the model

The model is trained on a training sample of the dataset. For semi-supervised node classification, we separate the data in a train:validate:test ratio of 20:20:60. The training of the model begins with creating the different layers using the number of nodes and features. The adjacency matrix is constructed from the graph, and the loss function used is the built in Categorical Cross-Entropy function. This loss function is used to evaluate the model. The data used to validate the results are taken by creating a tuple with the features data, the adjacency matrix, the labels and the boolean mask of the validation data. The features data and the adjacency matrix are then passed as the input data into mode.fit to train the model until the early stopping condition is reached.

### GCN Evaluation
To evaluate the model we will use the T-distributed Stochastic Neighbour Embedding (t-SNE) plot, as well as the accuracy and the loss of the model.

#### T-distributed Stochastic Neighbour Embedding (t-SNE)

t-SNE converts similarities between datapoints to probabilities and uses these to visualise high-dimensional data. The t-SNE embedding can be seen below. The grouping of data seen in the red lines indicates a high similarity between those data points, and as such we should expect to see a higher accuracy in the classification report for this category. However, the lines aren't continuous and the data points aren't necessarily grouped in the same location, so this could negatively affect the accuracy. Also category 3, coloured in red, is the category with the most notable grouping, whilst the others have significantly more variation. 

![tSNE](https://github.com/EvaHolden/PatternFlow/blob/topic-recognition/recognition/FacebookNetworkGCN/tsne.png?raw=true)

#### Accuracy
The research conducted by [S. Lui, J. Park, and S. Yoo](https://www.osti.gov/servlets/purl/1580233) concludes that an accuracy of greater than 60% for GCN is required for a model to be valuable. This model reaches this accuracy, as seen in the figure below, however an even greater accuracy could be obtained by further processing the data to remove uncorrelated data or by using a larger dataset to train on. Alternatively, the t-SNE plot shows data of all categories grouping, and the separations between the groups are independent of the category. This heavily implies that the categories chosen, of politicians, government organisations, television shows and companies, are not the best feature to categorise the nodes by and there is likely some other feature that the groups have in common that could be more accurate in classifying the nodes.

![Accuracy Plot](https://github.com/EvaHolden/PatternFlow/blob/topic-recognition/recognition/FacebookNetworkGCN/accuracy.png?raw=true)

#### Loss
The loss evaluates how much the predicted node classification deviates from the nodes actual classification. The figure below shows that the loss of the model is minimised, indicating that the node classification was accurate.

![Loss Plot](https://github.com/EvaHolden/PatternFlow/blob/topic-recognition/recognition/FacebookNetworkGCN/loss.png?raw=true)

### References
* https://www.osti.gov/servlets/purl/1580233 <br>
* https://www.tensorflow.org/<br>
* https://keras.io/<br>
* https://numpy.org/ <br>
* https://pandas.pydata.org/ <br>
* https://keras.io/ <br>
* https://networkx.org/ <br>
* https://matplotlib.org/<br>
* https://towardsdatascience.com/graph-convolutional-networks-deep-99d7fee5706f<br>
* https://medium.com/analytics-vidhya/getting-the-intuition-of-graph-neural-networks-a30a2c34280d<br>
* https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html
