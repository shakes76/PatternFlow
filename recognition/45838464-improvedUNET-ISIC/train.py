from dataset import preprocess_data, preprocess_masks
from modules import Improved_UNet
from tensorflow.keras.optimizers import Adam
import numpy as np


def training(datapaths, batch_size, epochs):
    """
    Trains the Improved UNet model based on the given data.
    datapaths is a list in the form:
    [train_data_path/*jpg, train_truth_path/*png, val_data_path/*jpg, val_truth_path/*png]
    Learning rate as per the paper.
    """

    # x_train = preprocess_data(datapaths[0])
    # y_train = preprocess_masks(datapaths[1])

    # x_val = preprocess_data(datapaths[2])
    # y_val = preprocess_masks(datapaths[3])
   
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_val = np.load('x_val.npy')
    y_val = np.load('y_val.npy')

    model = Improved_UNet()

    # fit the model with normal learning rate
    model.compile(optimizer = Adam(0.0005), loss = "cro", metrics=['accuracy', modules.DSC])

    history = model.fit(x_train, y_train,  validation_data= (x_val, y_val),
                            batch_size=batch_size,shuffle='True',epochs=epochs)


    return model, history 