# Graph Convolutional Network with Facebook Large Page-Page Network Dataset
## The Algorithm
Graph Convolution Networks (GCN) is a variation of the Convolutional Neural Networks (CNN) where it has been adapted to process graphs. By training the network, 
it is possible to check if there are any relations between one group to another by analysing its elements. This is relevant to social media companies such as Facebook
where they advertise different people and groups based on shared interests. A GCN averages a node and its surrouding neighbours which is done by calculating
the product of an adjacency matrix, features, and weight matrix. This creates a single graph convolution layer and can be stacked so that the product of 
aggregated values can be found.
## How it Works
The GCN model uses two graph convolution layers to achieve an accuracy above 80% for the Facebook Large Page-Page Network dataset. A relu activation layer is applied after
the first graph convolution layer followed by a dropout function to prevent co-adaption of feature detectors by randomly zeroing elements (a probability of 50% was used).
For training, the dataset was divided into three same length arrays for training, validating and testing. This was chosen due to its simplicity and can be changed in future
testing to improve accuracy and loss values. To train the GCN, negative log likelihood was used to calculate loss and the adam optimiser 
with a learning rate of 0.01 and weight decay of 0.0005 [1] was used. The network was then
trained for 20 epochs using the training datset in addition to the validation dataset loss and accuracy values being displayed.

The pre-processed [Facebook dataset](https://graphmining.ai/datasets/ptg/facebook.npz) was used to train and test the GCN. The following graph shows the accuracy 
and loss curves during training for the training dataset. The accuracy and loss for the test dataset; however, the loss and accuracy was 1.59 and 26.7% respectively,
displaying that there is overfitting in the training model.
###### Visualisation
![Training Plot](https://user-images.githubusercontent.com/63988751/197104211-b0c7509e-5201-4910-a41e-2773540382cc.png)

## Dependencies
- PyTorch 1.12.1
- Python 3.7.15
- NumPy 1.21.6
- SciPy 1.7.3
- pandas 1.3.5
- UMAP 0.5.3

## References
[1] [Kipf and Welling (2017), SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907v4.pdf)
