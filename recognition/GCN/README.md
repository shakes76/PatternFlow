## Semi-Supervised Node Classification on Facebook Page-Page Using GCNs
### The Problem
The dataset used is the partially processed Facebook Large Page-Page Network dataset, a page-page graph representation of Facebook websites; nodes model Facebook pages, while edges represent the likes intersecting sites. This GCN model uses semi-supervised node classification to determine the Facebook defined labels for each of the websites: (0) Politicians, (1) Governmental Organizations, (2) Television Shows and (3) Companies, based on 128 feature vectors taken from the website descriptions associated with the site's purpose. There are 22470 nodes, and 171002 edges, with nodes labels encoded between 0-3. Note that these were still passed through a label encoder in datasets.py.

#### Algorithm description 
Graph Convolutional Neural Networks aim to learn a pattern of signals in a given graph, given a feature matrix for the graph, and a representative structure of the graph as an adjacency matrix. GCNs are used in contexts where relationships in data cannot be mapped linearly, instead representing structure through adjacency matrices, and feature information through feature matrices. Multiclass node classification refers to when GCNs use this information and learn a mapping of nodes to an embedding space that resembles similarity in the graph, eventually outputting the predicted classes of the input nodes. Node information is averaged across nodes and their neighbours before feeding them into a neural network that returns a vector of probabilities. This problem is semi-supervised, as only a portion of the nodes that the model learns on are labelled.  

### How it Works:
GCNs work by having nodes pass features to their neighbours. These features must be aggregated in a way that captures the number of nodes that possess a given feature:

<img width="503" alt="image" src="https://user-images.githubusercontent.com/86597504/197330143-d2798cb0-f27e-4366-928f-9c13abe7e301.png">

This aggregation can be achieved through averaging neighbouring nodes' features. To calculate this, the feature values of each neighbouring node are required, along with the number of neighbouring nodes given a specific node to classify. Feature values can be obtained using the feature values matrix, while the number of neighbouring nodes can be obtained through using an adjacency matrix. GCN layers achieve this through the following computation, which will be explained in the next sections. 

<img width="393" alt="image" src="https://user-images.githubusercontent.com/86597504/197330430-e2dee6b2-16bc-4671-9946-35311c3ecd52.png">


##### Preprocessing Adjacency Matrix & Row Normalisation 
To learn features from the product of an adjacency matrix with node features, the network also needs to be identify the features of an input node-- this means that self loops needed to be added to the graph. Additionally, estimating the total value of the features in a given dataset must that highly connected graph nodes are overly represented in the adjacency matrix. As a result, the scaling of these input values to the network will be adversely impacted by the number of neighbours that they are connected to, resulting situations where the model gradient may vanish due to a smaller feature sum, or overblow as a result of highly large feature values. 

Normalising the rows of the adjacency matrix to sum to 1 can help mitigate this issue, as this ensures that neighbour counts are scaled between 0 and 1. This is performed by taking the inverse of the degree matrix's diagonal and multiplying it with the adjacency matrix. While normalising the feature matrix is not necessary, it is good practice, as this ensures that the product of the feature matrix and the adjacency matrix will be scaled between 0 and 1. 
The resulting operation is shown below, with D representing the degree matrix and A representing the adjacency matrix. 

<img width="66" alt="image" src="https://user-images.githubusercontent.com/86597504/197324602-502366dd-e6ee-4fd2-a153-145530ad02f4.png">

Operations performed between this normalised matrix and the feature matrix during training is mathematically equivalent to averaging a node's neighbours. Nodes that are only connected to one neighbour will have a stronger connection than nodes that are connected to multiple nodes in the graph. A problem with this approach is that it does not consider the number of neighbours that a given node's neighbour may have, which can be an issue if lower degree nodes represent more significant nodes in the problem. Performing symmetric normalisation through taking the normalised Laplacian matrix of the graph can solve this issue, as it multiplies the adjacency matrix by both the row and column normalised degree matrices- resulting in the below operation. 

<img width="122" alt="image" src="https://user-images.githubusercontent.com/86597504/197324623-7e0a4653-151f-4753-a2d6-cba5c4191d30.png">

Note that this model only performs row normalisation on the adjacency matrix, as symmetric row normalisation results in out of memory exceptions when testing as a result of the dataset's size. The model still achieves reasonable accuracy despite this change. 

##### Algorithm Steps:
Similar to other neural networks, the propagation of hidden node representations through a GCN can be summarised as follows: 
<img width="214" alt="image" src="https://user-images.githubusercontent.com/86597504/197328274-abe06da6-bf66-4a80-95b1-9d6cbe5dc011.png">

Where H(l + 1) refers to the value of a hidden layer,  A refers to the adjacency matrix, and f refers to activation rule. 

The activation rule predominantly involves applying an activation function over the matrix product of the normalised adjacency matrix, the feature matrix, and a weight tensor, as shown below:

<img width="127" alt="image" src="https://user-images.githubusercontent.com/86597504/197328760-23ab84a4-7766-4820-8d99-3802ae028946.png">

This corresponds to aggregating the feature information from the neighbours of each node in a sample neighbourhood. 

The amount of layers dictates how far feature vectors can travel between nodes: a 1 layer GCN means that each node is passed features from its immedate neighbours-- adding an extra layer allows each node to retrieve feature information an extra node away from its current list of neighbours. Adding a greater number of layers increases the amount of feature information that each node learns: however, with a higher number (>5) of layers, each node receives the aggregated features of the entire graph, which is not useful when trying learn similarities between nodes. Generally, best model performance is seen with GCNs of 2-3 layers.
A softmax function is applied over the final layer to return the probabilities of the number of classes passed. 

##### Semi-supervised learning aspect 


#### Using this file 
* Running train.py will train and test your model for the defined epochs, hidden layers and outputs selected. A run_training() function is responsible for performing these functions. If this is not desired, comment out the call to this function at the bottom of the file. 

* Predict.py demonstrates an example of the GCN performing node classification on a subset of nodes in the graph. It also produces a tSNE plot of the emebeddings generated from this subset, as well as from the whole graph. 

* Dataset.py will load and preprocess your inputs to provide normalised feature and adjacency matrices that are ready to be trained on. 

* Modules.py defines a Net model that defines the GCN, which connects to three GCN Layers. Each layer uses relu activation functions, with a final softmax layer for assigning probabilities to each of the classes. 


##### Dependencies required
* Python 3.8.12 used (though other versions may be suitable)
* Pytorch 1.12
* Networkx
* NumPy 
* Pandas
* Matplotlib
* Sklearn
* SciPy

##### Preprocessing Steps 
###### Train/Validation/Test Splits 
Training sets were split into 60%, leaving 20% for testing, and 20% for validation. This helped ensure that the validation set used to select the best performing model was to be just as representative of the entire dataset as the subset used to test the data with. 

#### Example Inputs 
Example inputs include loading in a subset of the given dataset and adjacency matrix (must be square). The model will return the classification of the input nodes, along with a tSNE plot of the neighbourhood embeddings, coloured by their ground truth labels. 

#### Performance Plots 
After 300 epochs of training, the following accuracy and loss curves are achieved. 

![Unknown-39](https://user-images.githubusercontent.com/86597504/197160327-f9185887-62ae-46b0-bdaf-67049310353e.png)

![Unknown-38](https://user-images.githubusercontent.com/86597504/197160334-ced94df3-c90b-45b9-9d43-88a223082ffb.png)


Hyperparameter tuning involved adjusting the number of hidden layers, along with the number of convolutional layers passed in. 
To decide on the ideal number of hyperparameters, the values 10, 15, 20 and 25 were trialled. 20 resulted in the highest validation performance of 70%, resulting in an overall test accuracy of 72%. 

To prevent overfitting, two dropout layers were also interspersed, with dropout values of 0.3 (i.e.: 30% of the nodes at each layer are dropped). Changing this value to 0.2 causes validation accuracy to drop to 67%.

#### tSNE Output 
The test output obtained from testing returns the following:

The following result is achieved as a result of plotting the model embeddings via tSNE:
![Unknown-40](https://user-images.githubusercontent.com/86597504/197188056-1b96fafd-60dd-4cc0-8c95-f3b04c76c835.png)

tSNE works by capturing structure in the input spaces and projecting them to a lower 2 dimensional space for interpretation. In this case, tSNE is displaying the learned embeddings of the GCN to a space that captures the similarities between nodes in the network.

As seen from the tSNE embeddings, the algorithm is most effective at correctly classifying the similarity of the label 3- "Television Shows", although there appears to be some overlap between the classifications of 3 and 2 ("Television Shows"), and Classes 1 ("Governmental Organisations") and 2. Class 0 appears to be the worst separated- almost entirely obscured from view, suggesting that pages from this class bear high feature similarity to other classes and are therefore listed as close in the embedding space. 


#### Sources: 
https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf
https://snap.stanford.edu/data/facebook-large-page-page-network.html
https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix
https://www.topbots.com/graph-convolutional-networks/
