# author: Yaoyu Liu 

import keras.backend as K

# do the Dice similarity coefficient function
def dice(X, Y, smooth=1):
    X_f = K.flatten(X)
    Y_f = K.flatten(Y)
    return (2.0 * K.sum(X_f * Y_f) + smooth) / (K.sum(X_f) + K.sum(Y_f) + smooth)
