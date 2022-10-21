# Model.py by Paul Turculetu (GCN algortihm)
# Feel free to use any of this code for any of your needs
# November 2021
# Final report
# Training a GCN for the facebook dataset and producing a tsne

import pandas as pd
import numpy as np
import stellargraph as sg
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
from tensorflow.keras import layers, optimizers, losses, Model
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def run_model():
    """loads and preprocess data, then it trains, evaluates and graphs a TSNE 
        on the classification of nodes. It also graphs the error on the evaluation
        and training dataset
    """

    # load the numpy arrays of the data given in the question
    # also find out how many classes the target variable has
    np_edges = np.load("edges.npy")
    np_features = np.load("features.npy")
    np_target = np.load("target.npy")
    
    # store the data as dataframes also, give the columns proper names
    # so things don't become confusion. Make data into a graph with edges and nodes
    df_features = pd.DataFrame(np_features)
    df_edges = pd.DataFrame(np_edges)
    df_targets = pd.DataFrame(np_target)
    df_edges.columns = ["source", "target"]
    df_targets.columns = ["target"]
    mat = sg.StellarGraph(df_features, df_edges)
    
    # split the data into train, test and validation keeping in my that 
    # the train and validation sets need to be significantly smaller than
    # the testing set.
    train_data, test_data = train_test_split(df_targets, train_size=500)
    val_data, test_data = train_test_split(test_data, train_size=500)
    
    # one hote encode the target datasets because right now each class is
    # represented by a string
    one_hot_target = pre.LabelBinarizer()
    train_targets = one_hot_target.fit_transform(train_data['target'])
    val_targets = one_hot_target.transform(val_data['target'])
    test_targets = one_hot_target.transform(test_data['target'])
    
    # initialize the model changing the hyper parameters to get
    # better results
    generator = FullBatchNodeGenerator(mat, method="gcn")
    train_gen = generator.flow(train_data.index, train_targets)
    gcn = GCN(
        layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator, dropout=0.2
    )
    x_in, x_out = gcn.in_out_tensors()
    pred = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    
    # optimize the model using the adam optimizer
    model = Model(inputs=x_in, outputs=pred)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    val_gen = generator.flow(val_data.index, val_targets)
    
    
    # train the model
    result = model.fit(
        train_gen,
        epochs=100,
        validation_data=val_gen,
        verbose=2,
        shuffle=False
    )
    
    # show an accuracy graph
    sg.utils.plot_history(result)
    
    # Test the model on the testing data
    test_gen = generator.flow(test_data.index, test_targets)
    print("testing data accuracy given below: ")
    model.evaluate(test_gen)
    
    # set up the tnse by getting the full dataset
    all_nodes = df_targets.index
    all_gen = generator.flow(all_nodes)
    
    embedding_model = Model(inputs=x_in, outputs=x_out)
    emb = embedding_model.predict(all_gen)
    X = emb.squeeze(0)
    
    # turn the data into 2 dimensions.
    tsne = TSNE(n_components=2)
    X_2 = tsne.fit_transform(X)
    
    # do an tsne plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_2[:, 0],X_2[:, 1],c=df_targets.squeeze(),cmap='turbo',
        alpha=0.5)
    ax.set(
        title="TSNE visualization of GCN embeddings for facebook dataset"
    )

