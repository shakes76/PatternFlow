
# GCN on Facebook.npz Dataset
### By: Arsh Upadhyaya
### roll_no: s4753993
## Goal 

The objective is to accurately classify nodes into there respective categories(total 4)
as per the facebook.npz dataset
## Why GCN

GCN and a conventional CNN are similar in there objectives, they both take some features 
from existing/given data and perform some operation on it, be it classification, recognition or even 
creation(like in GANs). However the difference lies in there flexibility. A CNN requires data in some 
clear format(like in images). However in a lot of real world applications, this is not the case. In fact, 
the internet itself is a graph with the websites themselves being the nodes and the edges being hyperlinks.
Such is the case with the facebook dataset, and we have to classify the websites to certain classes.

## Working of GCN

GCN use graphs as objects, and to create these, they multiply features by features by weights. They use adjacency matrix to store data. This matrix is used in forward pass(function used in model) and thus forms a graph, in which adjacent nodes have information about each other. For a simple 2 layer convolutional network, like the one used in the model, each node has information of nodes within 2 edges. Hence one can Imagine just how strong a deep GCN can be, as unlike the neurons in CNN, the nodes in GCN are not bound by euclidean geometry. Especially on a bigger dataset(internet), it could be assumed there would be a lot of hyperbolic and elliptical non-euclidean connections over space, thus increasing the interconnectivity of the model.

## Program Flow

#load_data()
this function is responsible for first extracting features,edges and targets, then create an adjacency matrix, and normalize that matrix.
Further it returns features as a tensor.

#GCN_model()
Since the given data has input of first layer as 128 dim vectors, a hidden layer of 32 was used and then directly finished with out_class=4, which is the number of required categories. Alternatives, a third layer could have also been added, however the accuracy was already pretty good.
Maybe a dropout was not necessary in hindsight as the dataset seems to small to need it.

train_model()
considered 200 epochs, and at each epoch, found difference between ouput of model and the target itself. 
Similarly for losses as well.
Done for both training and validation set.


## results
### accuracy plot
<img width="493" alt="Accuracy" src="https://user-images.githubusercontent.com/116279628/197387353-7d547ee3-5bdb-40e9-87af-0ed3f9322dc8.png">

we can see that training_accuracy>validation_accuracy, but only by a little bit which is a good sign.
Reached accuracy of 0.923 on training set after 200 epochs, went as high as 0.9446 after 400 epochs.

### loss plot
<img width="506" alt="loss" src="https://user-images.githubusercontent.com/116279628/197387403-62f1ecaf-7bff-4197-b324-64ffcaec5c64.png">

### test_set 
<img width="641" alt="test" src="https://user-images.githubusercontent.com/116279628/197387288-5faaf529-5733-4e6c-907a-de1bc56c757f.png">

The test_accuracy and test_loss is pretty similar to training and validation accuracy and losses as expected

### tsne
![GCN_tsne_200](https://user-images.githubusercontent.com/116279628/197387517-f2536959-50be-4cbb-b4a1-f1d5fb506dc5.png)

After 200 epochs there is a significant accuracy, and the 4 different classification categories are very rarely overlapping

## Dependancies
1.numpy

2.matplotlib

3.pytorch

4.python

5.sklearn

6.scipy

## reference
[1] https://arxiv.org/abs/1609.02907

[2]  https://github.com/tkipf/pygcn/tree/1600b5b748b3976413d1e307540ccc62605b4d6d/pygcn
especially helpful in training and testing functions
