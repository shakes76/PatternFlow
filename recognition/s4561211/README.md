# Graph Convolutional Network
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
Obtain the adjacency matrix from the information of Edges and normalize by node degree by multiplying adjacency matrix with the inverse degree matrix.

#### Data splits
For a semi supervised model, data is split into train, validation and test in the ratio of 20, 20 and 60 respectively.

### Model 
#### Graph Convolutional Network Model
It is a powerful neural network on graphs and can produce useful feature representations of nodes. <br>

GCN (graph convolutional network) is a neural network operating on graphs. <br>
Given a graph G = (V, E) as input where V is nodes and E is edges. <br>
* Input matrix: |V|* F⁰ where F⁰ is the number of input features and |V| is the number of nodes. <br>
* Adjacency matrix: |V|* |V| <br>
* Hidden layers Hⁱ = f(Hⁱ⁻¹, A): |V|* Fⁱ where H⁰ = V and f is a propagation. <br>

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

#### Usage
python facebook_GCN.py

### Outputs
Accuracy about node classification which is the prediction of Facebook sites <br>

#### Output for each epoch
        Epoch: 195 
        train - loss: tensor(0.4785, grad_fn=<NegBackward>) , accuracy tensor(0.8407) 
        validation - loss: tensor(0.4874, grad_fn=<NegBackward>) , accuracy tensor(0.8405) 
        -------------------------------------------------- 
        Epoch: 196
        train - loss: tensor(0.4763, grad_fn=<NegBackward>) , accuracy tensor(0.8469) 
        validation - loss: tensor(0.4962, grad_fn=<NegBackward>) , accuracy tensor(0.8376) 
        -------------------------------------------------- 
        Epoch: 197 
        train - loss: tensor(0.4634, grad_fn=<NegBackward>) , accuracy tensor(0.8402) 
        validation - loss: tensor(0.4852, grad_fn=<NegBackward>) , accuracy tensor(0.8382) 
        -------------------------------------------------- 
        Epoch: 198 
        train - loss: tensor(0.4671, grad_fn=<NegBackward>) , accuracy tensor(0.8482) 
        validation - loss: tensor(0.4791, grad_fn=<NegBackward>) , accuracy tensor(0.8431) 
        -------------------------------------------------- 
        Epoch: 199 
        train - loss: tensor(0.4735, grad_fn=<NegBackward>) , accuracy tensor(0.8402) 
        validation - loss: tensor(0.4688, grad_fn=<NegBackward>) , accuracy tensor(0.8425) 
        -------------------------------------------------- 

#### Accuracy for training and validation set
Both accuracies increase steadily so no overfitting happens. Furthermore, we got reasonable accuracy (0.8) after 150 iterations.

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
TSNE is t-Distributed Stochastic Neighbor Embedding. <br>
We can find that categories are clearly clustered in groups for outputs of first graph convolution layer compared to original features. <br>

#### original features
![](https://github.com/SandyKang/PatternFlow/raw/topic-recognition/recognition/s4561211/Resource/no_tsne.png) <br>
    
#### outputs of first graph convolution layer
![](https://github.com/SandyKang/PatternFlow/raw/topic-recognition/recognition/s4561211/Resource/tsne200.png) <br>
  
### References
[1] Kipf, Thomas N., and Max Welling. 2021. "Semi-Supervised Classification With Graph Convolutional Networks". Arxiv.Org. https://arxiv.org/abs/1609.02907. <br>
[2] Pathak, Manish. 2021. https://www.datacamp.com/community/tutorials/introduction-t-sne. <br>



    
