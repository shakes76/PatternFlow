
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow.keras as tfk
import numpy as np
from tensorflow.keras.datasets import cifar10 
from model import *
from tensorflow.keras.callbacks import CSVLogger

#Setting tensor values to float32 for memory usage, this is a deep network
K.set_floatx('float32')
print('Loading cifar10 data')
trainData, testData = cifar10.load_data()
n_classes = 10

#Separating training/test sets
x_train, y_train = trainData[0], to_categorical(trainData[1], num_classes=n_classes)
x_test, y_test = testData[0], to_categorical(testData[1], num_classes=n_classes)

x_shape = x_train.shape
n_samples, h, w, n_channels = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
print("samples %d \nh %d \nw %d \nchannels %d" % (n_samples, h, w, n_channels))
print('Normalising data')
x_train, x_test = normalise(x_train), normalise(x_test)
n_batch_size = 200
print('Creating efficientNet model')

#Hard coded hyperparameters and alpha-beta-gamma ratios compiled by the original model
argument_blocks = [
BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25), BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25), BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25), BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25), BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25), BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)]

size = x_train.shape
inputs = tfk.Input(shape=(size[1], size[2], size[3]))
X = inputs
classes = y_train.shape[1]
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

#Start adding all blocks
last_block = False
num_blocks = len(argument_blocks)
for i in range(num_blocks):
	last_block = True if (i+1 == num_blocks) else False
	X = mobile_conv_block(X, argument_blocks[i], i, classes, last_block)

#Callback for debugging and plotting data
logger = CSVLogger('./loss.csv', append=False)
model = tfk.Model(inputs=inputs, outputs=X, name='EfficientNet-7-block')

#print(model.summary())
model.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])
model.fit(x_train, y_train, batch_size=n_batch_size, epochs=500, validation_data=(x_test, y_test), verbose=0, callbacks=[logger])
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%"   %   (model.metrics_names[1],   scores[1]*100))
