## Semi-Supervised Node Classification on Facebook Page-Page Using GCNs
### Dataset Background
The dataset used is the partially processed Facebook Large Page-Page Network dataset, a page-page graph representation of Facebook websites; nodes model Facebook pages, while edges represent the likes intersecting sites. This GCN model uses semi-supervised node classification to determine the Facebook defined labels for each of the websites: (0) Politicians, (1) Governmental Organizations, (2) Television Shows and (3) Companies, based on 128 feature vectors taken from the website descriptions associated with the site's purpose. There are 22470 nodes, and 171002 edges, with nodes labels encoded between 0-3 [1]. Note that these were still passed through a label encoder in datasets.py.

#### Algorithm Description & Problem it Solved
Graph Convolutional Neural Networks aim to learn a pattern of signals in a given graph, given a feature matrix for the graph, and a representative structure of the graph as an adjacency matrix. GCNs are used in contexts where relationships in data cannot be mapped linearly, instead representing structure through adjacency matrices, and feature information through feature matrices [0]. Multiclass node classification refers to when GCNs use this information and learn a mapping of nodes to an embedding space that models similarity in the graph, eventually outputting the predicted classes of the input nodes. This problem is semi-supervised, as only a portion of the nodes that the model learns on are labelled.  

### How it Works:
GCNs work by having nodes pass features to their neighbours. These features must be aggregated in a way that captures the number of nodes that possess a given feature:

<img width="503" alt="image" src="https://user-images.githubusercontent.com/86597504/197330143-d2798cb0-f27e-4366-928f-9c13abe7e301.png">

This aggregation can be achieved through averaging neighbouring nodes' features. To calculate this, the feature values of each neighbouring node are required, along with the number of neighbouring nodes. Feature values can be obtained using the feature values matrix, while the number of neighbouring nodes can be obtained through using an adjacency matrix. GCN layers achieve this through the following computation, which will be explained in the next sections. 

<img width="393" alt="image" src="https://user-images.githubusercontent.com/86597504/197330430-e2dee6b2-16bc-4671-9946-35311c3ecd52.png">


##### Preprocessing Adjacency Matrix & Row Normalisation 
Taking the product of the adjacency and feature matrix helps obtain an average of neighbouring node's features. Before this can be achieved, however, the adjacency matrix must be pre-processed. 

Adjacency matrices do not contain self loops, but a feature representation of the node itself still needs to be passed to its neighbours-- this means that self loops needed to be added to the graph. Highly connected graph nodes are also over-represented in the adjacency matrix. As a result, the scaling of these input values to the network will be adversely impacted by the number of neighbours that they are connected to, resulting situations where the model gradient may vanish due to a smaller feature sum, or overblow as a result of large feature values. 

Normalising the rows of the adjacency matrix to sum to 1 can help mitigate this issue, as this ensures that neighbour counts are scaled between 0 and 1. This is performed by taking the inverse of the degree matrix's diagonal and multiplying it with the adjacency matrix. While normalising the feature matrix is not necessary, it is good practice, as this ensures that the product of the feature matrix and the adjacency matrix will be scaled between 0 and 1. 

The resulting operation is shown below, with D representing the degree matrix and A representing the adjacency matrix:

<img width="66" alt="image" src="https://user-images.githubusercontent.com/86597504/197324602-502366dd-e6ee-4fd2-a153-145530ad02f4.png">

Multiplying this normalised matrix and the feature matrix during training is now mathematically equivalent to averaging a node's neighbours. Nodes that are only connected to one neighbour will have a stronger connection than nodes that are connected to multiple nodes in the graph. A problem with this approach is that it does not consider the number of neighbours that a given node's neighbour may have, which can be an issue if lower degree nodes represent more significant nodes in the problem [2]. 

Performing symmetric normalisation through taking the normalised Laplacian matrix of the graph can solve this issue, as it multiplies the adjacency matrix by both the row and column normalised degree matrices- resulting in the below operation:

<img width="122" alt="image" src="https://user-images.githubusercontent.com/86597504/197324623-7e0a4653-151f-4753-a2d6-cba5c4191d30.png">

Note that this model only performs row normalisation on the adjacency matrix, as symmetric row normalisation results in out of memory exceptions when testing as a result of the dataset's size. The model still achieves reasonable accuracy despite this change. 

##### Algorithm Steps:
Similar to other neural networks, the propagation of hidden node representations through a GCN can be summarised as follows:

<img width="214" alt="image" src="https://user-images.githubusercontent.com/86597504/197328274-abe06da6-bf66-4a80-95b1-9d6cbe5dc011.png">

Where H(l + 1) refers to the value of a hidden layer,  A refers to the adjacency matrix, and f refers to activation rule. 

The activation rule predominantly involves applying an activation function over the matrix product of the normalised adjacency matrix, the feature matrix, and a weight tensor, as shown below:

<img width="127" alt="image" src="https://user-images.githubusercontent.com/86597504/197328760-23ab84a4-7766-4820-8d99-3802ae028946.png">

This corresponds to aggregating the feature information from the neighbours of each node in a sample neighbourhood. At each layer of the GCN, this aggregation is performed on each node. 

The amount of layers dictates how far feature vectors can travel between nodes: a 1 layer GCN means that each node is passed features from its immedate neighbours-- adding an extra layer allows each node to retrieve feature information an extra node away from its current list of neighbours. Adding a greater number of layers increases the amount of feature information that each node learns: however, with a higher number (>5) of layers, each node receives the aggregated features of almost the entire graph, which is not useful when trying learn similarities between nodes. Generally, best model performance is seen with GCNs of 2-3 layers [3].

A softmax function is applied over the final layer to return the probabilities of the number of classes passed. 

##### Semi-supervised Learning Aspect 
The model creates an initial embedding on the entire dataset, but since the loss function is only defined on the training indices when training the model, and the validation subset when validating the model, the model only learns on a sample of labelled data. This approach ensures a more resilient model: exposure to unlabelled data encourages the model to learn the structure of the network, while the subset of labelled data allows the model to refine and develop predictions based on a source of truth. 

#### How to Use This Submission:
##### General Notes
* Ensure that the facebook.npz dataset is downloaded in the same directory as the working directory for your environment, as file paths are currently set to be your relative working directory. The dataset can be downloaded here: https://graphmining.ai/datasets/ptg/facebook.npz

* Hyperparameter constants such as the number of hidden layers and classes can be found at the top of modules.py, while constants required for training such as the number of epochs can be found at the top of training.py. 

##### Breakdown of files 
* Running train.py will train and test your model for the defined epochs, hidden layers and outputs selected. A run_training() function is responsible for performing these functions. If this is not desired, comment out the call to this function at the bottom of the file. 

* Predict.py demonstrates an example of the GCN performing node classification on a subset of nodes in the graph. It also produces a tSNE plot of the emebeddings generated from this subset, as well as from the whole graph. 

* Dataset.py will load and preprocess your input data to provide normalised feature and adjacency matrices that are ready to be trained on. 

* Modules.py defines a Net model that defines the GCN, which connects to three GCN Layers. Each layer uses relu activation functions, with a final softmax layer for assigning probabilities to each of the classes. 


##### Dependencies Required
* Python 3.8.12 used (though other versions may be suitable)
* Pytorch 1.12
* Networkx
* NumPy 
* Pandas
* Matplotlib
* Sklearn
* SciPy

##### Model Summary:
The GCN network is represented by a Net class that takes in feature column size, hidden layers and desired outputs as input parameters, and requires input data and a square adjacency matrix for a forward pass through the network. Each layer in the network is represented by the GNNLayer class, which performs a matrix multiplication of the feature and weight matrix, and a sparse matrix multiplication with the adjacency matrix before applying an activation function. 

The following summary breaks down the model's structure. 
* 1 x GNNLayer (128, 20)
* 1 x Dropout Layer (DROPOUT = 0.3)
* 1 x (linear) GNNLayer (20, 20) 
* 1 x Dropout Layer (DROPOUT = 0.3)
* 1 x GNNLayer (20, 4)
* 1 X softmax activation function applied to last layer 

The following summarises a GNN Layer:
* input, output features as defined in Net class
* Uniform Xavier initialisation of a weight matrix with same input output dimensions as GNNLayer. Xavier initialisation initialises weights such that inputs and outputs from a layer have the same amount of variance, helping prevent exploding or vanishing gradients.
* matrix multiplication of features and weight 
* sparse multiplication of above product with adjacency matrix
* activation function applied to output. 

##### Preprocessing Steps 
###### Train/Validation/Test Splits 
Training sets were split into 60%, leaving 20% for testing, and 20% for validation. This helped ensure that the validation set used to select the best performing model was to be just as representative of the entire dataset as the subset used to test the data with. 

#### Example Inputs 
Example inputs include loading in a subset of the given dataset and adjacency matrix (must be square). The model will return the classification of the input nodes, along with a tSNE plot of the neighbourhood embeddings, coloured by their ground truth labels:

<img width="683" alt="image" src="https://user-images.githubusercontent.com/86597504/197389764-dda2fe47-b85e-46ac-826c-125316271dbe.png">


#### Experimentation 
Hyperparameter tuning involved adjusting the number of hidden layers, and the amount of dropout performed in the network.

To decide on the ideal number of layers, the values 10, 15, and 20 were trialled. To prevent overfitting, two dropout layers were also interspersed between the fully connected layers, with dropout values of 0.3 (i.e.: 30% of the nodes at each layer are dropped). 

The following experimental results were achieved by modifying the N_HID constant defined at the top of predict.py accordingly. 

10 layers resulted in validation accuracy of 55%, and a training accuracy of 56%. The following figure indicate that no further changes in both training and validation accuracy were seen from roughly 125 epochs onwards, suggesting an increase in model complexity was likely required. 

![image](https://user-images.githubusercontent.com/86597504/197387284-69a0e3e0-8526-473f-b6f6-eeee723b7fae.png)

15 layers resulted in validation accuracy of 74% and a training accuracy of 74%.

![image](https://user-images.githubusercontent.com/86597504/197388129-b4ad446d-faa8-4f1f-83ac-0cf428da19a0.png)

20 resulted in the highest training and validation performance of 77%, resulting in an overall test accuracy of 76%. As seen from the plot in the performance plots subheading, the accuracy declines from epochs 50-100, before consistently increasing. Training performance does not taper off after 300 epochs, suggesting that overall performance could improve with further training. Additionally, loss consistently decreases, suggesting that although accuracy fluctuates initially, the scale of these errors is not large. 

![Unknown-39](https://user-images.githubusercontent.com/86597504/197160327-f9185887-62ae-46b0-bdaf-67049310353e.png)

![Unknown-38](https://user-images.githubusercontent.com/86597504/197160334-ced94df3-c90b-45b9-9d43-88a223082ffb.png)


To perform the following experiments on dropout, modify the DROPOUT constant defined at the top of modules.py to the desired dropout value. 

DROPOUT was set to 0.5, a commonly used value in literature [4]. The following figure demonstrates that validation performance outperformed training performance, with a decline in model performance between 50-100 epochs, resulting in a 75% and a 76% training and validation accuracy. 

![image](https://user-images.githubusercontent.com/86597504/197391898-7ed55ce4-66cb-4b8e-9e57-822be8c028d4.png)

Changing DROPOUT to 0.4  results in 74% and 76% training and validation accuracy, with validation accuracy outperforming training accuracy. Overall model performance is more stable than it is at DROPOUT = 0.5, consistently increasing with each epoch:
![image](https://user-images.githubusercontent.com/86597504/197392963-5b8cee77-db63-4bd7-9365-f0df9cd70aee.png)

Nevertheless, the model with DROPOUT = 0.3 was chosen, as the validation accuracy for this model was marginally higher. 

#### Performance Plots 
After 300 epochs of training, the following accuracy and loss curves are achieved. 

![Unknown-39](https://user-images.githubusercontent.com/86597504/197160327-f9185887-62ae-46b0-bdaf-67049310353e.png)

![Unknown-38](https://user-images.githubusercontent.com/86597504/197160334-ced94df3-c90b-45b9-9d43-88a223082ffb.png)


#### Example Outputs
When using predict.py on a subset of the data (not for testing purposes), the predicted classes of input nodes are returned, along with the tSNE plot of the model's embeddings: 

<img width="401" alt="image" src="https://user-images.githubusercontent.com/86597504/197384946-005c6f4e-9fdd-4657-9bb4-5c6112a5f5eb.png">

The training and validation loss and accuracy can be seen at each epoch when training the model: 

<img width="1103" alt="image" src="https://user-images.githubusercontent.com/86597504/197385048-fee798a8-5adc-45c8-8858-d37edcdb8c71.png">

The  output obtained from testing the model returns the overall test accuracy and loss:

<img width="621" alt="image" src="https://user-images.githubusercontent.com/86597504/197384981-dee9004f-5516-4ea9-b7f4-d7f7677b9e98.png">

##### tSNE Output 
The following result is achieved as a result of plotting the model embeddings via tSNE with 2 components, and colouring points with their ground truth labels:

![Unknown-40](https://user-images.githubusercontent.com/86597504/197188056-1b96fafd-60dd-4cc0-8c95-f3b04c76c835.png)

tSNE works by mapping similarity between pairs of points to probability distributions in the original dimensions and a low dimensional space, then minimising the divergence between these two distributions to provide the optimal 2D representation of the similarities between points.

In this case, tSNE is displaying the learned embeddings of the GCN to a space that captures the similarities between nodes in the network-- points that cluster together are recognised as closer neighbours in the network. Ideally, the classes would be well-separated with low within class distance and high between class distance. This would suggest that the embeddings learned only associate websites of the same class as similar. 

As seen from the tSNE embeddings, the algorithm is most effective at correctly classifying the similarity of the label 3- "Television Shows", although there appears to be some overlap between the classifications of 3 and 2 ("Television Shows"), and Classes 1 ("Governmental Organisations") and 2. Class 0 appears to be the worst separated- almost entirely obscured from view, suggesting that pages from this class may bear high feature similarity to other classes and are therefore listed as close in the embedding space. 

#### Sources: 
* [0] https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book-Chapter_5-GNNs.pdf
* [1] https://snap.stanford.edu/data/facebook-large-page-page-network.html
* [2] https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix
* [3] https://www.topbots.com/graph-convolutional-networks/
* [4] https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
