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
![GCN layers picture](readMeImage/gcnLayer.png)
## Reference
[1] https://arxiv.org/abs/1609.02907
