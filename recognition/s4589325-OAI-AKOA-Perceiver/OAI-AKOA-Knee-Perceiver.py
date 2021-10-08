#!/bin/python3

# ##### Setup #####

# Libraries needed for model
import tensorflow as tf
from tensorflow.keras import layers, models
import PIL

# Libraries needed for data importing
import os
import itertools
from PIL import Image
import math
import numpy as np # just need this for a single array conversion in the preprocessing step - please don't roast me

#print("Tensorflow Version:", tf.__version__)

# ##### Macros #####
SAVE_DATA			= False
BATCH_SIZE			= 32
TEST_TRAINING_SPLIT	= 0.8
IMG_WIDTH			= 260
IMG_HEIGHT			= 228
SEED				= 123
LATENT_SIZE			= 512 # Same as paper
LEARNING_RATE		= 0.001
WEIGHT_DECAY		= 0.0001
EPOCHS				= 50
DROPOUT_RATE		= 0.2
PATCH_DIM			= 2

# ##### Import Data #####

def save_data():
	dataDirectory = '../../../AKOA_Analysis/'

	# Need to sort data by patient so that we aren't leaking data between training and validation sets
	allPics = [dataDirectory + f for f in os.listdir(dataDirectory)]

	patients = [[0] * 2 for _ in range(len(allPics))]

	i = 0
	for pic in allPics:
		# Get unique id for each patient
		pic = pic.split('OAI')[1]
		baseline_num_str = pic.split('de3d1')[0].split('BaseLine')[1]
		baseline_num = int(''.join(c for c in baseline_num_str if c.isdigit()))
		pic_num = pic.split('.nii.gz_')[1].split('.')[0]
		initial_id = pic.split('_')[0] + '_BaseLine_' + str(baseline_num) + '.' + pic_num
		patients[i][0] = initial_id
		i = i + 1

	# Assign each patient left or right status (slow code)
	ii = 0
	for i in patients:
		if ('right' in allPics[ii].lower()) or ('R_I_G_H_T' in allPics[ii]):
			patients[ii][1] = 1
		else:
			patients[ii][1] = 0
		ii += 1

	print('Right Knees:', sum([i[1] for i in patients]))

	# Sort by substring
	patients = [list(i) for j, i in itertools.groupby(sorted(patients))]

	print('Patients:', len(patients))

	patients = patients[0:math.floor(len(patients) * 0.1)]

	# Split data
	patients_train	= patients[0:math.floor(len(patients) * TEST_TRAINING_SPLIT)]
	patents_test	= patients[math.floor(len(patients) * TEST_TRAINING_SPLIT):-1]

	# Remove extra axis added by the above sorting line
	patients_train	= [item for sublist in patients_train for item in sublist]
	patients_test	= [item for sublist in patents_test for item in sublist]

	# Import/Load images
	xtrain = []
	ytrain = []
	for i in patients_train:
		for j in allPics:
			if i[0].split('.')[0] in j and i[0].split('.')[1] in j.split('de3')[1]:
				xtrain.append(np.asarray(PIL.Image.open(j).convert("L")))
				ytrain = i[1]
				break
	xtest = []
	ytest = []
	for i in patients_test:
		for j in allPics:
			if i[0].split('.')[0] in j and i[0].split('.')[1] in j.split('de3')[1]:
				xtest.append(np.asarray(PIL.Image.open(j).convert("L")))
				ytest = i[1]
				break

	# Normalize the data to [0,1]
	xtrain = np.array(xtrain, dtype=float)
	xtest  = np.array(xtest, dtype=float)
	xtrain[:] /= 255
	xtest[:] /= 255

	# Save the data to local drive
	np.save('../../../xtrain', xtrain)
	np.save('../../../ytrain', ytrain)
	np.save('../../../xtest', xtest)
	np.save('../../../ytest', ytest)

# Save the data
if SAVE_DATA:
	save_data()

# Load Data
print("Loading Data")
#xtrain = np.load('../../../xtrain.npy')
#ytrain = np.load('../../../ytrain.npy')
xtest = np.load('../../../xtest.npy')
#ytest = np.load('../../../ytest.npy')

print(xtest.shape)

''' # Cannot use this code because it leaks data between training/test sets
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
	validation_split=0.2,
	subset="training",
	seed=SEED,
	label_mode=None,
	image_size=(IMG_WIDTH, IMG_HEIGHT),
	batch_size=BATCH_SIZE,
	color_mode='grayscale'
)
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
	validation_split=0.2,
	subset="validation",
	seed=SEED,
	label_mode=None,
	image_size=(IMG_WIDTH, IMG_HEIGHT),
	batch_size=BATCH_SIZE,
	color_mode='grayscale'
)
'''

def pos_encoding(img, bands, Fs):
	# Create grid
	n, x_size, y_size = img.shape
	x = tf.linspace(-1, 1, x_size)
	y = tf.linspace(-1, 1, y_size)
	xy_mesh = tf.meshgrid(x,y)
	xy_mesh = tf.transpose(xy_mesh)
	xy_mesh = tf.expand_dims(xy_mesh, -1)
	xy_mesh = tf.reshape(xy_mesh, [x_size,y_size,2,1])
	# Frequency logspace of nyquist f for bands
	up_lim = tf.math.log(Fs/2)/tf.math.log(10.)
	low_lim = math.log(1)
	f_sin = tf.math.sin(tf.experimental.numpy.logspace(low_lim, up_lim, bands) * math.pi)
	f_cos = tf.math.cos(tf.experimental.numpy.logspace(low_lim, up_lim, bands) * math.pi)
	t = tf.concat([f_sin, f_cos], axis=0)
	t = tf.concat([t, [1.]], axis=0) # Size is now 2K+1
	# Get encoding/features
	encoding = xy_mesh * t
	encoding = tf.reshape(encoding, [1, x_size, y_size, (2 * bands + 1) * 2])
	encoding = tf.repeat(encoding, n, 0) # Repeat for all images (on first axis)
	out = tf.concat([img, encoding], axis=-1)
	out = tf.reshape(out, [n, x_size*y_size, -1]) # Linearise (doesn't work!)
	print(out)
	print(out.shape)
	out = tf.reshape(out, [n, x_size * y_size, -1])
	return out

pos_encoding(xtest, 4, 500)


# ##### Define Modules #####
