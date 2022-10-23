
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
## Process of solving problem
## results
### accuracy plot
<img width="493" alt="Accuracy" src="https://user-images.githubusercontent.com/116279628/197387353-7d547ee3-5bdb-40e9-87af-0ed3f9322dc8.png">

we can see that training_accuracy>validation_accuracy, but only by a little bit which is a good sign

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
