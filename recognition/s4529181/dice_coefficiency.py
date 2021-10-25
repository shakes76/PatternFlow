# author: Yaoyu Liu 

import keras.backend as K
import numpy as np


# do the Dice similarity coefficient function
def dice(X, Y, smooth=1):
    X_f = K.flatten(X)
    Y_f = K.flatten(Y)
    return (2.0 * K.sum(X_f * Y_f) + smooth) / (K.sum(X_f) + K.sum(Y_f) + smooth)

# the Dice similarity coefficient for numpy
def dice_np(X, Y, smooth=1):
    X_f = X.flatten(X)
    Y_f = Y.flatten(Y)
    return (2.0 * np.sum(X_f * Y_f) + smooth) / (np.sum(X_f) + np.sum(Y_f) + smooth)
