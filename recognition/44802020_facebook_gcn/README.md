# Facebook Graph Convolutional Neural Network

**Name**: John Parsons

**Student Number**: 44802020

**Student Email**: john.parsons@uqconnect.edu.au

**Task**: Facebook Large Page-Page Network Dataset (Task 2)

* * *
### Contents
* [Introduction to the Problem and the Dataset](#Introduction to the Problem and the Dataset)<br>
* [The Algorithm](#The Algorithm)<br>
* [Project Structure](#Project Structure)<br>
* [Running the Model and Dependencies](#Running the Model and Dependencies)<br>
* [Data and Training](#Data and Training)<br>
* [Building the Model](#Building the Model)<br>
* [Compiling the Model](#Compiling the Model)<br>
* [Performance and Analysis](#Performance and Analysis)<br>
* * *

### Introduction to the Problem and the Dataset

The data is a connected graph of Facebook pages which can each be categorised
as 1 of 4 types of pages (TV Shows, Companies, Government Agencies or 
Politicians). The model is a Graph Convolutional Neural Network (GCN) which 
aims to be able to categorise a given page into one of these 4 categories. The 
data set contains 22470 nodes. 

Using SciKitLearn's TSNE analysis, the data generates the following TSNE plot:

![TSNE_Train](./resources/TSNE_Plot_(Train%20Data).png)

### The Algorithm

The algorithm revolves around taking advantage of the fact that any given node
may be likely to be of the same category as its neighbours.

We take the normalized Adjacency Matrix A_bar (where all nodes are self connected)
and multiply it by the Feature Matrix and by the Weights Matrix. The result is then
run through an activation function (I used softmax) and the loss is calculated with 
a loss function (I used Sparse Categorical Cross Entropy). The then model tries to 
minimize this loss with an optimization function (I used Adam) and adjusts the 
Weights Matrices accordingly.

Overtime, hopefully, the losses will be minimized by optimal weights and the model
will become more accurate.

###  Project Structure

There are two `.py` files in the project as well as a `resources` folder:
- `myGraphModel.py`: This file contains the code involved in creating the 
Model. This consists of one class (`FaceGCNLayer`, which represents my 
custom network layer) and one function (`makeMyModel`, which creates a 
model which represents mt GCN and contains all the relevant layers).

- `driver.py`: This fie is responsible for parsing the data, calling `makeMyModel`
from `myGraphModel.py`, parsing and splitting the data in training/validation sets,
running the model, tracking and displaying the progress/accuracy of the model and 
plotting a TSNE plot of the data.
- `resources`: This is where you should put the `facebook.npz` file. This is
also where the images embedded in this file are stored.

### Running the Model and Dependencies

To run the model, run `driver.py.main()`. Ensure that the `FILE_PATH` variable is set
to the location of the `facebook.npz` file. The `facebook.npz` file will need to be 
downloaded to an appropriate local location. by default the `FILE_PATH` variable points
to the `resources` folder in the same directory as the `driver.py` file. The user can 
easily adjust the following
model variables prior to running:
- `PLOT_TSNE`: Set whether you want to plot accuracy.
- `PLOT_ACCURACY`: Set whether you want to plot accuracy.
- `EPOCHS`: Set the number of epochs over which the Model should train.
- `LEARNING_RATE`: Set the Model learning rate.
- `TRAIN_SPLIT`: The portion of the data to split into the training set.
- `TEST_VAL_SPLIT`: The portion of the data to split into the test/validation set.

The user should also ensure that the following **dependencies** are installed and up to date:

- Tensorflow 2.6.0
- Keras 2.6.0
- Scipy 1.7.1
- Numpy 1.19.5
- Sklearn 1.0.1
- Matplotlib 3.4.3

### Data and Training 

The dataset has a stated density of 0.001, making it a very sparse graph.
This is ideal for use of a Tensorflow Sparse Tensor to improve performance.

An 80:20 Training:Testing/Validation split was used. This is because the fast nature
of the model (Using Sparse Tensors and multiple Dense layers to reduce 
dimensionality, as well as a relatively small dataset) mean it would be 
ideal for the model to be trained as large a portion of the data as possible.

It is however, as was found in testing, very easy for a GCN model to over-fit to
data. It was therefore not feasible to split the data 90:10, or something similar, 
as it was important that the model's accuracy could be validated on  a large
testing/validation set to ensure it is not over-fitted.

###  Building the Model

The model is built by calling the `makeMyModel` function in `myGraphModel.py`.
This function creates a `tensorflow.keras.models.Sequential` object and adds 
the following layers to it, where N is the number of nodes (varies between 
training and testing/validation).
* * *
| Layer (type)                      | Output Shape | Param # | 
| --------------------------------- | ------------ | ------- |
| `face_gcn_layer (FaceGCNLayer)`   | (N, 128)     | 128     |
| `dropout (Dropout)`               | (N, 128)     | 0       |
| `dense (Dense)`                   | (N, 64)      | 8256    |
| `face_gcn_layer_1 (FaceGCNLayer)` | (N, 64)      | 64      |
| `dropout_1 (Dropout)`             | (N, 64)      | 0       |
| `dense_1 (Dense)`                 | (N, 32)      | 2080    |
| `face_gcn_layer_2 (FaceGCNLayer)` | (N, 32)      | 32      |
| `dropout_2 (Dropout)`             | (N, 32)      | 0       |
| `dense_2 (Dense)`                 | (N, 4)       | 132     |
* * *
The model uses 4 categories of layers:
- **FaceGCNLayer Layers**: This is the layer responsible for the computation.
- **Reduction Dense Layers**: These layers exist to reduce the dimensionality 
of the data to allow more epochs to be run more quickly.
- **Dropout Layers**: These layers randomly drop a portion of the weights to 0.
This helps to avoid over-fitting.
- **Final Dense Layer**: This layer is required to categorise each node into 
one of 4 categories using 'softmax' activation.

###  Compiling the Model

**Optimizer:**  For optimization I used Sparse Categorical Cross Entropy. 
Categorical Cross Entropy is used to calculate loss when dealing with multiple
categories. Sparse Categorical Cross Entropy is the same, only it allows for
the use of Sparse Tensors.

**Loss Function:** Adam, the default Keras model loss function was used. 
this is because while other Loss functions were tested (including Nadam 
and Adamax), none yielded any significant improvement in terms of accuracy.

### Performance and Analysis

The model reaches around 72% Validated Accuracy with 200 epochs, 
plateauing at this value at around 30 epochs.

![Learning_200](./resources/Learning_(200).png)

The data plateau is likely due to the model over-fitting. I tried to remedy
this by using Dropout layers, as these would randomly eliminate some 
weights responsible for the over-fitting. THis did help initially as the 
model was plateauing at around 53%, but the model still plateaus at 72%

As can be seen from the following TSNE plots (one of the training data 
one of the testing data), while there are pockets that are clearly segmented, 
the data overall is not very neatly segmented. This indicates that the 
different categories share many similar features, making it difficult to 
accurately categorise the nodes.

![TSNE_Train](./resources/TSNE_Plot_(Train%20Data).png)

![TSNE_Train](./resources/TSNE_Plot_(Test%20Data).png)

This difficulty makes the 72% accuracy achieved somewhat reasonable, although 
it may be further improved. The inclusion of Skip Connections could help, although 
it is not possible to do with a Sequential Model and would require the model to be 
redesigned in a non-sequential manner. 

Like-wise, the Sequential model does not allow
multiple arguments to be passed to the layers. This means that the layer behaviour 
can only change based on whether the layer is training or testing but not on whether
it is testing or validating (This is why the data is only split into Train/Testing
and not Train/Validate/Test).

In summary, redesigning the model in a non-sequential form would allow for several 
avenues of potential improvement. 

