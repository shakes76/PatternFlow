# Multilayer- GCN
GCNs are nothing more than graph convolutional networks. This just refers to the type of data imported. In the case of CNN usually there is a 2D image and a filter
and sequentially the filter moves through the image doing operations on what it sees. But now assume that your data is a graph where nodes are connect to each other
via an edge. The reason why CNNs would not work in this case is because the graph representing the dataset is not part of a Euclidean space. This makes CNNs useless
for datasets represented by graphs. In short, GCNs are just CNNs for sets of data represented with a graph. An example of such a dataset (and the dataset I will
make a GCN on) is the [Facebook Large Page-Page Network dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html). I will classify the nodes of
this dataset, that means that given the **128 of the features** of each webpage, I will predict whether or not they are part of one of the four classes:
* tvshow
* government 
* company 
* politician

Given that this problem is supposed to be semi supervised, the dataset needs to be split accordingly. This means that the training and validation sets need to be
significantly smaller than the testing set.


## How it works
1. The program imports and reads the data given as numpy arrays which are then changed to pandas dataframes
2. The data is split into training validation and testing and because it is semi supervised the training and validation sets only have 500 points in it
3. A graph like data structure is created from the set of data
4. One hot encoding is used to get the categorical target variable to a numerical state
5. The model is then initialised using arbitrary hyperparameters at first but later it is tuned for better accuracy
6. The model is then trained and evaluated
7. The evaluation consists of a graph showing the validation and training accuracy and loss for each epoch.
8. Using the training data the model tries to predict the page outcome. 
9. The model is then shrunk down to 2 dimensions and plotted using tsne

## Train and validation graphs
![train and validation accuracy](https://raw.githubusercontent.com/Pentaflouride/PatternFlow/topic-recognition/recognition/45249435/train_val%20accuracy.png)

