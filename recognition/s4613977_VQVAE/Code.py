from os import listdir
from os.path import isfile, join
import numpy as np
import imageio
import matplotlib.pyplot as plt

#First loading all images into numpy arrays
#Note that this also normalises the data

def load_data():
    X_train=[]
    for f in listdir('train'):
        path='train/'+f
        X_train.append(list(imageio.imread(path)))
    X_train=np.array(X_train)

    X_test=[]
    for f in listdir('test'):
        path='test/'+f
        X_test.append(list(imageio.imread(path)))
    X_test=np.array(X_test)

    X_val=[]
    for f in listdir('val'):
        path='val/'+f
        X_val.append(list(imageio.imread(path)))
    X_val=np.array(X_val)

    X_train = X_train[:,:,:,np.newaxis]
    X_test = X_test[:,:,:,np.newaxis]
    X_val = X_val[:,:,:,np.newaxis]

    X_train = X_train.astype(float)/255.
    X_test = X_test.astype(float)/255.
    X_val = X_val.astype(float)/255.
    
    return X_train, X_test, X_val

#Sanity check
X_train, X_test, X_val = load_data()

print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("X_val shape:",X_val.shape)
