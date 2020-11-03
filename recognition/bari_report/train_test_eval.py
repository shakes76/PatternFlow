"""
Created on Oct 30, 2020

@author: s4542006, Md Abdul Bari
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
from pre_process import data_part_normal
from unet import my_unet


def get_dsc(y_pred, y_true):
    """calculate both class-wise and overall DSC"""
    axes = (1, 2) # widtha and height axes of each image
    intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    union = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    # tiny fraction is added to avoid division by zero error
    smooth = 1e-10
    dice = 2 * (intersection + smooth)/(union + smooth)
    # overall and class-specific DSC
    dice_class = np.mean(dice, axis=0)
    dice_overall = np.mean(dice)
    return (dice_class, dice_overall)


def model_train_test(n_epochs=50, batch_size=32):
    """compile, fit UNet model and return model training history and test DSC"""
    print("\nCurrent epoch number to train the model is: {}".format(n_epochs))
    bool_epoch = input("\nDo you want to change the epoch number? (y/n): ")
    bool_epoch = bool_epoch.lower()
    if bool_epoch is None:
        n_epochs = n_epochs
    elif bool_epoch[0] == 'y':
        n_epochs = int(input("\nProvide the desired number of epochs: "))
    else:
        n_epochs = n_epochs
    data_dict = data_part_normal()
    X_train, y_train = data_dict["train"][0], data_dict["train"][1]
    X_val, y_val = data_dict["validation"][0], data_dict["validation"][1]
    X_test, y_test = data_dict["test"][0], data_dict["test"][1]
    y_test_cat = to_categorical(y_test)
    model = my_unet()
    model.summary()
    model.compile(optimizer=Adam(), loss="binary_crossentropy", 
                  metrics=["accuracy"])
    # early stopping conditions and saving best model
    callbacks = [EarlyStopping(patience=10, verbose=1),
                 ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, 
                                   verbose=1),
                 ModelCheckpoint('model-isic-unet.h5', verbose=1, 
                                 save_best_only=True, save_weights_only=True)]
    # model fitting
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, \
              callbacks=callbacks, validation_data=(X_val, y_val))   
    #loading best model weigth parameters 
    model.load_weights('model-isic-unet.h5')
    model.evaluate(X_test, y_test, verbose=1)
    pred_test_prob = model.predict(X_test, verbose=1)
    # assign class label baosed on threshold probability (0.50)
    pred_test_class = (pred_test_prob > 0.50).astype(np.uint8)
    pred_test_cat = to_categorical(pred_test_class)
    class_dsc = get_dsc(pred_test_cat, y_test_cat)[0]
    overall_dsc = get_dsc(pred_test_cat, y_test_cat)[1]
    return ({"class_dsc": class_dsc, "overall_dsc": overall_dsc, 
            "input_test": X_test, "gt_test": y_test, 
            "pred_test": pred_test_class}, {"history": history})


def get_loss_plot(hist=None):
    """create plot of training and validation loss"""
    plt.figure(figsize=(10, 5))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.plot( np.argmin(hist.history["val_loss"]), 
             np.min(hist.history["val_loss"]), 
             marker="x", color="r", label="Best model")
    plt.xlabel("Number of epochs")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.show();
    
    
def plot_sample_image(X, y_actual, y_pred, ix=None):
    """plot input, actual and ground truth image from the test set"""
    if ix is None:
        ix = random.randint(0, len(X))
    else:
        ix = ix
    fig, ax = plt.subplots(1, 3, figsize=(25, 15))
    
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].contour(X[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Input Image')

    ax[1].imshow(y_actual[ix, :,:,0], cmap='gray')
    ax[1].contour(y_actual[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Ground Truth')

    ax[2].imshow(y_pred[ix, ..., 0], cmap='gray')
    ax[2].contour(y_pred[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted Image')
    plt.show()
    
    
if __name__ == "__main__":
    print("This module provides utility functions training and testing UNET",
          "and is not meant to be executed on its own.")