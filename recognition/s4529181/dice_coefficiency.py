# author: Yaoyu Liu 

import keras.backend as K

# do the Dice similarity coefficient function
def dice(X, Y):
    X_f = K.flatten(X)
    Y_f = K.flatten(Y)
    return (2. * K.sum(X_f * Y_f)) / (K.sum(X_f) + K.sum(Y_f)) 