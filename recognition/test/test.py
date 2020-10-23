'''
test infor
'''

import tensorflow as tf
from tensorflow.keras.datasets import mnist
print(tf.__version__) 

#parameters
epoches = 2 # multiple of 3 +1
batch_size = 64

#load data
print("> Loading images ...")
#loads the training, and testing data (ignoring class labels)
(x_train, _),(x_test, y_test) = mnist.load_data()
#scales the training and test data to range between 0 and 1
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()
#x_train = (x_train - 127.5) / 127.5 normalize to [-1,1]
#x_test = (x_test - 127.5) / 127.5 normalize the image to [-1,1]
x_train = x_train.reshape((len(x_train), 28,28,1))
x_test = x_test.reshape((len(x_test), 28,28,1))
total_training, xSize, ySize, c = x_train.shape
print(x_train.shape)

#layers
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(total_training).batch(batch_size)

#build networks