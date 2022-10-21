## Semi-Supervised Node Classification on Facebook Page-Page Using GCNs
#### Algorithm description 
Graph Convolutional Neural Networks aim to learn a pattern of signals in a given graph, given a feature matrix for the graph, and a representative structure of the graph as an adjacency matrix. GCNs are used in contexts where relationships in data cannot be mapped linearly, instead representing structure through adjacency matrices, and feature information through feature matrices. Multiclass node classification refers to when GCNs use the learned relationships between its inputs to eventually outputting the predicted classes of the input nodes. This problem is semi-supervised, as only a portion of the nodes that the model learns on are labelled. 

### The Problem
The dataset used is the partially processed Facebook Large Page-Page Network dataset, a page-page graph representation of Facebook websites; nodes model Facebook pages, while edges represent the likes intersecting sites. This model aims to determine the Facebook defined labels for each of the websites: (0) Politicians, (1) Governmental Organizations, (3) Television Shows and (4) Companies, based on 128 feature vectors taken from the website descriptions associated with the site's purpose. There are 22470 nodes, and 171002 edges, with numerically labelled nodes. 


### GCN vs CNN
The aggregation performed by GCN is achieved through averaging neighbouring nodes' features. To calculate this, the feature values of each neighbouring node are required, along with the number of neighbouring nodes given a specific node to classify. Feature values can be obtained using the feature values matrix, while the number of neighbouring nodes can be obtained through using an adjacency matrix. The multiplication of these two matrices therefore provides an estimate of the total of all the features in a given neighbourhood. 

#### Steps:
##### Preprocessing Adjacency Matrix & Row Normalisation 
To learn features from the product of an adjacency matrix with node features, the network also needs to be identify the features of an input node-- this means that self loops needed to be added to the graph. Additionally, estimating the total value of the features in a given dataset must that highly connected graph nodes are overly represented in the adjacency matrix. As a result, the scaling of these input values to the network will be adversely impacted by the number of neighbours that they are connected to, resulting situations where the model gradient may vanish due to a smaller feature sum, or overblow as a result of highly large feature values. 

Normalising the rows of the adjacency matrix to sum to 1 can help mitigate this issue, as this ensured that the product of the adjacency and feature matrices are now scaled between 0 and 1. This is performed by taking the inverse of the degree matrix's diagonal and multiplying it with the adjacency matrix. Operations performed between this normalised matrix and the feature matrix during training is mathematically equivalent to averaging the graph's neighbours. 

Performing symmetric normalisation ensures that features can be weighted according to their importance. For example, a feature with an average degree 


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
Training sets were split into 60%, leaving 20% for testing, and 20% for validation. This helped ensure that the validation set used to select the best performing model was to be just as representative of a random subset used to test the data with. 

#### Example Inputs 
Example inputs include loading in a subset of the given dataset and adjacency matrix (must be square). The model will return the classification of the input nodes, along with a tSNE plot of the neighbourhood embeddings, coloured by their ground truth labels. 

#### Performance Plots 
After 300 epochs of training, the following accuracy and loss curves are achieved. 

![Unknown-39](https://user-images.githubusercontent.com/86597504/197160327-f9185887-62ae-46b0-bdaf-67049310353e.png)

![Unknown-38](https://user-images.githubusercontent.com/86597504/197160334-ced94df3-c90b-45b9-9d43-88a223082ffb.png)


Hyperparameter tuning involved adjusting the number of hidden layers, along with the number of convolutional layers passed in. 
To decide on the ideal number of hyperparameters, the values 10, 15, 20 and 25 were trialled. 20 resulted in the highest validation performance of 70%, resulting in a test accuracy of 72%. 

To prevent overfitting, two dropout layers were also interspersed, with dropout values of 0.3 (i.e.: 30% of the nodes at each layer are dropped). Changing this value to 0.2 causes validation accuracy to drop to 67%.

#### tSNE Output 
The following result is achieved as a result of plotting the model embeddings via tSNE:
![Unknown-40](https://user-images.githubusercontent.com/86597504/197188056-1b96fafd-60dd-4cc0-8c95-f3b04c76c835.png)


1. The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.
3. provide example inputs, outputs and plots of your algorithm
5. Describe any specific pre-processing you have used with references if any.

#### Sources: 
https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf
https://snap.stanford.edu/data/facebook-large-page-page-network.html
