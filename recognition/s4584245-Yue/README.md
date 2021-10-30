# COMP3710

## s4584245 / YueXiang Zhao


## Graph convolution network introduction

Many problems are graphs in nature. In the real world, we see a lot of data are graphs, such as molecules, social networks, and knowledge graph(Google).This project aims to create a suitable multi-layer GCN model to carry out a semi supervised multi-class node classification using Facebook Large Page-Page Network dataset(https://snap.stanford.edu/data/facebook-large-page-page-network.html) with reasonable accuracy(about 0.9).GCNs are a very powerful neural network for deep learning on graphs. In fact, the most indepensable part in GCN is that a randomly initiated 2-layer GCN can generate feature representations of nodes in networks.In this case, I will use GCN model to process the facebook data set.
The following figure shows the principle of GCN.

<img src="https://raw.githubusercontent.com/yuexiangzh/PatternFlow/topic-recognition/recognition/s4584245-Yue/gcn.png">

## Data set

Facebook Large Page-Page Network Dataset statistics

- Nodes: 22,470

- Edges: 171,002 

- Node features: Yes.

- Categories:  politicians, governmental organizations, television shows and companies.

## Train parameters and procedure
- Load dataset:
I have to load the data and prepare the data. In order to reduce the project difficulty, I used the facebook.npz to get the features,edges and target.In this part,I set up three functions(load data, normalizeadj, normalize features) to prodeuce the adjacency and return to some important parameters.I also split the data into three different parts.After that, we can write the GCN model layers.
- GCN layer:
Based on the code I construct the two GCN layers. I use two layers of GCN layer for the GCN model. The input dimension of first layer is 128 (128 dim vectors) and output dimension is 16. The input dimension of second layer is 16 and output dimension is 4 (4 categories:politicians, governmental organizations, television).
- GCN model:
Lastly, train the model and use loss_acc function to get the val accuracy about 0.9. Users can run the test_script.py document to get the accuracy and tsne result.


## How to run model
- Download all files in github file.
- Put files in a same document.
- Download the requirement packages.
- Open and run test_driver_script.py

## Output
- 1.The range of accuracy is from 0.3145 to 0.8885
```
=====================================================
Epoch 000: Loss 1.6843, TrainAcc 0.304, ValAcc 0.3145
Epoch 001: Loss 1.4315, TrainAcc 0.3155, ValAcc 0.3330
Epoch 002: Loss 1.3371, TrainAcc 0.3751, ValAcc 0.3990
Epoch 003: Loss 1.3003, TrainAcc 0.5192, ValAcc 0.5400
Epoch 004: Loss 1.2577, TrainAcc 0.5946, ValAcc 0.6075
Epoch 005: Loss 1.2118, TrainAcc 0.587, ValAcc 0.5940
Epoch 006: Loss 1.1733, TrainAcc 0.6131, ValAcc 0.6200
Epoch 007: Loss 1.1294, TrainAcc 0.6565, ValAcc 0.6665
Epoch 008: Loss 1.0801, TrainAcc 0.6748, ValAcc 0.6945
Epoch 009: Loss 1.0258, TrainAcc 0.6686, ValAcc 0.6770
Epoch 010: Loss 0.9710, TrainAcc 0.6615, ValAcc 0.6710
Epoch 011: Loss 0.9196, TrainAcc 0.687, ValAcc 0.6985
Epoch 012: Loss 0.8688, TrainAcc 0.716, ValAcc 0.7260
Epoch 013: Loss 0.8227, TrainAcc 0.7337, ValAcc 0.7445
Epoch 014: Loss 0.7830, TrainAcc 0.7618, ValAcc 0.7675
Epoch 015: Loss 0.7335, TrainAcc 0.7701, ValAcc 0.7735
Epoch 016: Loss 0.6986, TrainAcc 0.7879, ValAcc 0.7880
Epoch 017: Loss 0.6619, TrainAcc 0.8037, ValAcc 0.8010
Epoch 018: Loss 0.6246, TrainAcc 0.8072, ValAcc 0.8065
Epoch 019: Loss 0.5991, TrainAcc 0.8189, ValAcc 0.8175
...
```
- 2.The following figure shows the val-accuracy of the change with the number of epoch(150).
<img src="https://raw.githubusercontent.com/yuexiangzh/PatternFlow/topic-recognition/recognition/s4584245-Yue/acc.png">

- 3.According to the TSNE results, the data is roughly divided into four categories(4 colors)
<img src="https://raw.githubusercontent.com/yuexiangzh/PatternFlow/topic-recognition/recognition/s4584245-Yue/tsne.png">

## Requirement package
- pytorch
- numpy
- Matplotlib
- sklearn
- scipy
