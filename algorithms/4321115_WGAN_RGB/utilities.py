import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow import keras

"""
Custom constraint for the Wasser-Stein loss
"""

class ClipConstraint(Constraint):
	def __init__(self, clip_value):
		self.clip_value = clip_value
	
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)
	
	def get_config(self):
		return {'clip_value': self.clip_value}

import cv2
from os import listdir, mkdir, makedirs
from os.path import exists
import numpy as np
from numpy.random import randn
from numpy import expand_dims
from matplotlib  import pyplot as plot

"""
Loads the celeba dataset, checks width/height and adjusts all images to be 128x128x3, in [-1.0:1.0] domain
"""

def load_data(path="./celeba-dataset/img_align_celeba/img_align_celeba/", n_samples=1000, im_ref_size=256):
	file_exists = exists(path)
	assert(file_exists, "dataset does not exist: Please download the celeba dataset under this directory, and change the root directory name as celeba-dataset")
	files = np.random.choice(np.array(listdir(path)), n_samples, replace=False)
	ref_img = cv2.imread(path+files[0])
	w = int((im_ref_size - ref_img.shape[0])/2)
	h = int((im_ref_size - ref_img.shape[1])/2)
	new_size = int(im_ref_size/2)
	dim = (new_size, new_size)
	X = np.array([np.array(cv2.resize(cv2.copyMakeBorder(cv2.imread(path+f), w-1, w+1, h-1, h+1, cv2.BORDER_REPLICATE), dim)) for f in files])
	X = X.astype('float32')
	X = (X-127.5)/127.5
	return X

"""
Takes random non-repeated number o real images and returns both images and their classification (in this case -1.0 for real images)
"""

def generate_real_samples(X, n_samples=200):
	size = X.shape[0]
	indices = np.random.choice(size, n_samples, replace=False)
	samples = np.take(X, indices, axis=0)
	Y = -np.ones((samples.shape[0], 1))
	return samples, Y


"""
Generates fake images from a latent space and returns both reshaped fake image and y values (1.0 for fake)
"""

def generate_samples(model, n_samples=100):
	X_input = generate_latent_points(n_samples)
	X = model.predict(X_input)
	Y = np.ones((n_samples, 1))
	return X, Y

def generate_latent_points(n_samples, vector_points=100):
	X = randn(n_samples*vector_points)
	X = X.reshape(n_samples, vector_points)
	return X

"""
The Wasser-Stein loss used in the WGAN model
"""

def wasserStein_loss(Y_real, Y_predict):
	return K.mean(Y_real*Y_predict)

"""
Saves the final evaluation metrics and a line plot of the training error for the critic on real images, critic on fake images and the overall WGAN network loss
"""

def save_history(loss1, loss2, lossGan, plots):
	if plots:
		plot.plot(loss1, label='critic loss')
		plot.plot(loss2, label='generator critic loss 2')
		plot.plot(lossGan, label='gan loss')
		plot.legend()
		plot.savefig('./plot.jpg')
		plot.close()
		return
	type = ''
	fileExists = exists('metrics.csv')
	if not fileExists:
		type = 'w'
	else:
		type = 'a'
	with open('metrics.csv', type) as file:
		writer = csv.writer(file, delimiter=',')
		if not fileExists:
			writer.writerow(['epoch', 'critic loss', 'critic2 loss', 'gan loss'])
		for i in range(len(loss1)):
			writer.writerow([str(i+1), loss1[i], loss2[i], lossGan[i]])

"""
Saves a generated image at the given epoch, and saves a check-point model file.
"""

def summarise_performance(step, generator, laten_dim=100, n_samples=100):
	if not exists('./images'):
		mkdir('./images')
	X, Y = generate_samples(generator, 1)
	X = ((X+1)/2.0)*255
	fileName = 'generated_%04d.jpg' % (step)
	cv2.imwrite('./images/'+fileName, X[0])
	if not exists('./model'):
		mkdir('./model')
	fileName_gan = 'model%04d.h5' % (step+1)
	generator.save('./gan_model/'+fileName_gan)
