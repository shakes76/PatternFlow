'''
MNIST InfoVAE with MMD

@author Shakes
'''
import tensorflow as tf 
from tensorflow.keras.datasets import mnist

print('TensorFlow version:', tf.__version__)

#parameters
epochs = 2 #multiple of 3 + 1
batch_size = 64

#load data
print("> Loading images ...")
# Loads the training and test data sets (ignoring class labels)
(x_train, _), (x_test, y_test) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()
#x_train = (x_train - 127.5) / 127.5 # Normalize the images to [-1, 1]
#x_test = (x_test - 127.5) / 127.5 # Normalize the images to [-1, 1]
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))
total_training, xSize, ySize, c = x_train.shape
print(x_train.shape)

#layers


#Build networks


#Build models


#train


#test


#plot

print('END')
