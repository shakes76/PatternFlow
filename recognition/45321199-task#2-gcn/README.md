# Problem #2 - GCN model to identify different webpages in the Facebook dataset.

## Problem Statement (From Task Sheet):
Create a suitable multi-layer GCN model to carry out a semi supervised multi-class node classification using Facebook Large Page-Page Network dataset with reasonable accuracy. Also include a TSNE embeddings plot with ground truth in colors.

## Dataset Contents
This project uses the partially-processed 128-dim vectors, provided via "https://graphmining.ai/datasets/ptg/facebook.npz".

Contents:
    - Label Types:    [0 1 2 3]
    - Edges:          171002
    - Vertices:       22470
    - Features:       128

Where the label types represent a category of webpage; being politician, government institute, tv show or company.

## Usage
Before running "driver.py", ensure the latest version of the following modules are installed:
    - tensorflow
    - numpy
    - matplotlib
    - scipy
    - sklearn

"driver.py", upon running, will begin the full process of retrieving the dataset, processing it, creating the model, training it and procuring statistics/predictions.
Folder, "saved_model" will contain the model as a .h5 file, as well as the training history as a numpy matrix.
Folder, "figures" will contain all the plots including accuracy, loss and TSNE. Provided in this folder are also screenshots manually taken of the model structure, training results and predictions.

The entire process took roughly 1hr on an RTX 3060.

## The Process
The model structure in question is a typical GCN. Consisting of an input layer, 2 middle layers for additional complexity (relu), and an output layer (softmax).
Parameters (adjustable constants at the top of modules.py & train.py), are currently set to what is believed to generate a steady learning rate per epoch.
This resulted in the following model:
![Model](figures\model.png)

This model was then trained for 3000 epochs:
![Training](figures\training.png)

## Results
predict.py will then load the saved history & model from "saved_model", and automatically generate the following plots each run:

### Accuracy
![Accuracy](figures\acc.png)

### Loss
![Loss](figures\loss.png)

### Predictions
![Predictions](figures\predictions.png)

### TSNE Plot
![TSNE](figures\TSNE.png)

As seen in accuracy/loss, the model is closely fitted. This indicates that we have more than likely derived the full possible accuracy from the dataset (90%+). Extending or augmenting the dataset would therefore be the next step in improving the model's performance.

## References
Salama, K. (2021) Keras documentation: Node classification with Graph Neural Networks, Keras. Available at: https://keras.io/examples/graph/gnn_citations/ (Accessed: October 21, 2022). 

Mayachita, I. (2020) Training graph convolutional networks on node classification task, Medium. Towards Data Science. Available at: https://towardsdatascience.com/graph-convolutional-networks-on-node-classification-2b6bbec1d042 (Accessed: October 21, 2022). 