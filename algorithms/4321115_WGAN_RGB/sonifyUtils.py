import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from tensorflow.keras.layers import AveragePooling3D
from os.path import isdir
from os import makedirs

#Utilities for creating varying resolutions of a tensor for sound output

def averagePool(tensor, stride):
	poolSize = int(np.shape(tensor.numpy())[0]/stride)
	shape = np.shape(tensor.numpy())
	Z = tf.reshape(tensor, (1, shape[0], shape[1], 1))
	averagedPool = tf.nn.pool(Z, (stride, stride), strides=(stride, stride), pooling_type="AVG", padding="VALID")
	averagePool = np.reshape(averagedPool.numpy(), (poolSize, poolSize))
	return averagePool

def averagePool3D(tensor, factor):
	shape = np.shape(tensor)
	Z = tf.reshape(tensor, (1, shape[0], shape[1], shape[2], 1))
	Z = AveragePooling3D((factor, factor, 1))(Z)
	Z.numpy().shape
	Z = np.reshape(Z.numpy(), (int(shape[0]/factor), int(shape[1]/factor), 3))
	return Z

def divisors(size):
	print('Checking divisors for ', str(size))
	output = []
	for i in range(1, size):
		if size % i == 0 and i > 10 and size/i > 10:
			output.append(i)
	return output

def divisors2D(factors, size):
	print('Checking divisors for ', str(size))
	output = []
	for i in range(1, factors):
		if size[0] % i == 0 and size[1]%i == 0 and i > 5 and size[0]/i > 10:
			output.append(i)
	return output

def getHueScores(array):
	dict = {}
	dict[0] = 0
	dict[1] = 0
	dict[2] = 0
	for c in array:
		for r in c:
			dict[np.argmax(r)] += 1
	return np.flip(np.argsort([dict[n] for n in dict]))

def exportImage(image, name, path):
	if not isdir(path):
		print('Creating directory ', path)
		makedirs(path)
	plt.imshow(image)
	plt.tight_layout()
	plt.savefig(path+name+'.jpg')
