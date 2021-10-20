# Graph Gonvolutional Network
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
    
#### Data splits
For a semi supervised model, data is split into train, validation and test in the ratio of 20, 20 and 60 respectively.

### Model 
#### Graph Convolutional Network Model
![](https://github.com/SandyKang/PatternFlow/raw//topic-recognition/s4561211/Resource/GCN_model.png)

    
