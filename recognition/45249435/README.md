# Multilayer- GCN
GCNs are nothing more than graph convolutional networks. This just refers to the type of data imported. In the case of CNN usually there is a 2D image and a filter
and sequentially the filter moves through the image doing operations on what it sees. But now assume that your data is a graph where nodes are connect to each other
via an edge. The reason why CNNs would not work in this case is because the graph representing the dataset is not part of a Euclidean space. This makes CNNs useless
for datasets represented by graphs. In short, GCNs are just CNNs for sets of data represented with a graph. 
