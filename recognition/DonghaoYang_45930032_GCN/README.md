# Multi-layer GCN model to carry out a semi supervised multi-class node classification using Facebook Large Page-Page Network dataset
Author: Donghao Yang 45930032
## Algorithm and problem description
### GCN algorithm introduction
As we know, the input of traditional CNN is a graph structure with a Euclidean structure. However, more general graph structure we encounter a lot is 
social networks or topological networks. This kind of graph structure is irregular since there're different number of adjacency nodes or edges for each 
nodes, which means CNN cannot work on it. Luckily, Graph Convolutional Network appears and can solve this difficulties. The application of GCN includes 
recommendation system, natural language processing, computer vision and biochemisry. If GCN works on the edges of the graph, then it can conduct edge 
prediction. If GCN works on the nodes of the graph, then it can conduct nodes classification, which is the functionality I am going to realize in this 
report.
### The introduction of the theory behind GCN
From the picture below, the graph with C input channels is transferred to a graph with F channels after a multi layers of GCN operations. Channels here 
forms the features' matrix of the GCN input. Another input is the adjacency matrix of the graph.
![GCN layers picture](readMeImage/gcnLayer.png)
The below picture is the core mathematical theory behind the GCN. In a word, it represents the information of every node of the next layer is obtained 
by weighting and summing the information of the previous layer itself and its neighboring nodes. After that, conduct linear and nonlinear transformations. 
Let's explain the meaning of symbols one by one. A(hat) is actually A+I(A is the adjacency matrix of the graph, and I is the identity matrix of the graph) 
. By doing this, we can reserve the information of the nodes themselves when calculating. D^(-1/2)A(hat)D^(-1/2) is the symmetric normalization of A(hat) 
in case the scale of output of each GCN layer increases. Why we do like this is related to the Laplacian Matrix. H(l) is the feature matrix of the graph 
in layer "l". D^(-1/2)A(hat)D^(-1/2)H(l) is just matrix multiplication and then conducting a linear transformation by a weighting matrix W(l). After that, 
we get the input feature matrix of the next layer by a non-linear activation function.
![GCN Equation picture](readMeImage/GCN_equation.png)
### Facebook pages classification problem
The Facebook pages classification problem requires carry out a semi supervised multi-class node classification using [Facebook Large Page-page Network 
dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html). There are four classes for pages: politicians, governmental organization, 
television shows and companies. Firstly, the dataset needs to be preprocessed so that it's adhere to the requirements of the GCN train model. This can 
be done by the dataset.py. Next, we need to build a reasonable multilayer GCN model for this classification problem. This is done by the modules.py and 
our GCN model has 4 layers. Then, we need to train our defined multilayer GCN model with suitable metric and optimizer while we validate our model with 
validation data set. The loss and accuracy of the training and validating process should be drawn to some plots. This is done by the train.py. In this 
file, we conduct a test to our model with test dataset and draw a TSNE graph to showcase clusters as well. At last, we pack our trained model in a 
format can be used by user directly and showcase example usage of my trained model.
## Dependencies required
Running this GCN model to solve the Facebook pages classification problem, The below libraries and versions of python and tensorflow are required:<br />
[1]numpy;<br />[2]scipy;<br />[3]sklearn;<br />[4]keras;<br />[5]matplotlib;<br />[6]tensorflow2.9.2(GPU version);<br />[7]Python3.9.13 <br /><br />
The address of the required dataset is:
[Facebook Large Page-page Network 
dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html)
## Reference
[1] https://arxiv.org/abs/1609.02907 <br />
[2] https://snap.stanford.edu/data/facebook-large-page-page-network.html <br />
[3] https://github.com/ElonQuasimodoYoung/COMP3710_Report_Donghao_Yang_45930032_PatternFlow/tree/topic-recognition/recognition/FacebookNetworkGCN <br />
[4] https://github.com/ElonQuasimodoYoung/COMP3710_Report_Donghao_Yang_45930032_PatternFlow/tree/topic-recognition/recognition/s4521842_GCN/GCN <br />
[5] https://github.com/ElonQuasimodoYoung/COMP3710_Report_Donghao_Yang_45930032_PatternFlow/tree/topic-recognition/recognition/s4628887_Task_2_GCN <br />
