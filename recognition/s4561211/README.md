# Graph Gonvolutional Network
### Ya-Yu Kang

### Objective
The aim is to create a multi-layer graph convolutional network model to implement node classification with multi-class using dataset - Facebook Large Page-Page Network to receive reasonable accuracy and plot a TSNE embedding with labels.

### Dataset
#### Facebook Large Page-Page Network
This dataset is a page-page graph of Facebook sites collected via Facebook Graph API in November 2017. Each row is a node which is a Facebook page, and its features are from the site descriptions. As for edges, it shows the linkage between Facebook sites. There are 4 categories of sites defined by Facebook, politicians, governmental organizations, television shows and companies. 

    Details – original dataset
    --------------------------
    Nodes	22,470
    Edges	171,002
    --------------------------
    
    Partially processed dataset
    ----------------------------------------
    Nodes	22,470 with features 128 dimesons
    Edges	342,004 
    ----------------------------------------

#### Data pre-processing
Obtain the adjacency matrix from the information of Edges and normalize it

#### Data splits
For a semi supervised model, data is split into train, validation and test in the ratio of 20, 20 and 60 respectively.

### Model 
#### Graph Convolutional Network Model
It is a powerful neural network on graphs and can produce useful feature representations of nodes. <br>

This is my Graph Convolutional Network model: <br>

![](https://github.com/SandyKang/PatternFlow/raw/topic-recognition/recognition/s4561211/Resource/GCN_model.png) <br>

* 1st graph Convolutional layer: 128*32 <br>
* Activation Function: ReLU
* Dropout: 0.5
* 2nd graph Convolutional layer: 32*4 
* Activation Function: log softmax <br>

Loss functions: log loss <br>

Optimizer: Adam optimizer with learning rate 0.01 <br>

Best Model Selection: Save the best model which has the largest accuracy for validation data <br>

#### List of Packages Required
* numpy
* scipy.sparse
* torch
* sample
* imatplotlib
* sklearn
* seaborn

### Outputs
Accuracy for training and validation set <br>
Both accuracies increase steadily so no overfitting happens.

200 iterations: <br>
![](https://github.com/SandyKang/PatternFlow/raw/topic-recognition/recognition/s4561211/Resource/accuracy200.png) <br>

500 iterations: <br>
![](https://github.com/SandyKang/PatternFlow/raw/topic-recognition/recognition/s4561211/Resource/accuracy500.png) <br>

#### Accuracy for test set <br>
200 iterations: <br>
test - loss: tensor(0.4834, grad_fn=<NegBackward>) , accuracy tensor(0.8432) <br>
    
500 iterations: <br>
test - loss: tensor(0.3858, grad_fn=<NegBackward>) , accuracy tensor(0.8830) <br>

#### TSNE embedding with true labels
t-Distributed Stochastic Neighbor Embedding (outputs of first graph convolution layer) <br>
We can find that categories are clearly clustered in groups. <br>
    
![](https://github.com/SandyKang/PatternFlow/raw/topic-recognition/recognition/s4561211/Resource/tsne200.png) <br>
   
### References
[1] Kipf, Thomas N., and Max Welling. 2021. "Semi-Supervised Classification With Graph Convolutional Networks". Arxiv.Org. https://arxiv.org/abs/1609.02907. <br>
[2] Pathak, Manish. 2021. https://www.datacamp.com/community/tutorials/introduction-t-sne. <br>



    
