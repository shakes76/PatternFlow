import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
from tensorflow import keras
from keras import layers


hidden_units = [64, 64]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256


def setup_training(target):
    train_data = []
    test_data = []
    for _, group_data in target.groupby("page_type"):
        random_group = np.random.rand(len(group_data.index)) <= 0.8
        train_data.append(group_data[random_group])
        test_data.append(group_data[~random_group])
    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    return train_data, test_data


def handle_experiment(model, x_train, y_train):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )
    return history


def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


def create_ffn(name=None):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers, name=name)


def train_model(train_data, test_data, target, class_idx, page_name_idx):
    feature_names = set(target.columns) - {"id", "page_type"}
    num_features = len(feature_names)
    num_classes = len(class_idx)

    # print(page_name_idx)
    print("yes")
    x_train = train_data[feature_names].to_numpy()
    x_test = test_data[feature_names].to_numpy()
    y_train = train_data["page_type"]
    y_test = test_data["page_type"]

    """
    x_train = np.array(train_data[feature_names])
    x_test = np.array(test_data[feature_names])
    y_train = np.array(train_data["page_type"])
    y_test = test_data["page_type"]
    """
    baseline_model = create_baseline_model(num_classes, num_features)
    baseline_model.summary()
    history = handle_experiment(baseline_model, x_train, y_train)


def create_baseline_model(num_classes, num_features):
    inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])

    logit = layers.Dense(num_classes, name="logit")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logit, name="baseline")


def handle_training(sorted_data):
    edges, features, target = sorted_data
    train_data, test_data = setup_training(target)
    train_model(train_data, test_data, target)

