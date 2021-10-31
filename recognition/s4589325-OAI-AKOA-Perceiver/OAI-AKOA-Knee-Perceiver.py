#!/bin/python3

# General References:
# - https://keras.io/examples/vision/perceiver_image_classification/ (good resource to see general structure)
# - https://github.com/Rishit-dagli/Perceiver (decent for seeing how some functions are implemented)
# - https://keras.io/api/layers/base_layer/ (how to use classes with tf)

# ##### Setup #####

# Libraries needed for model
import tensorflow as tf
import tensorflow.keras
import tensorflow_addons as tfa
import PIL

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Libraries needed for data importing
import os
import itertools
from PIL import Image
import math
import numpy as np # just need this for a single array conversion in the preprocessing step - please don't roast me
import random
import copy

print("Tensorflow Version:", tf.__version__)
tf.config.run_functions_eagerly(True)

# ##### Macros #####
SAVE_DATA			= True
BATCH				= 8
TEST_TRAINING_SPLIT	= 0.75
IMG_WIDTH			= int(260 / 1)
IMG_HEIGHT			= int(228 / 1)
SEED				= 132
BANDS				= 6
MAX_FREQUENCY		= 10
VALIDATION_SPLIT	= 0.2
EPOCHS				= 1
INPUT_SHAPE			= (IMG_WIDTH, IMG_HEIGHT)
LATENT_ARRAY_SIZE	= 256 # Paper uses 512
BYTE_ARRAY_SIZE		= IMG_HEIGHT * IMG_WIDTH
CHANNEL				= (2 * BANDS + 1)
PROJECTION			= 2 * (2 * BANDS + 1) + 1
QKV_DIM				= PROJECTION
LEARNING_RATE		= 0.001
OPTIMIZER_WEIGHT	= 0.0001
DROPOUT_RATE		= 0.2
TRANSFOMER_NUM		= 6
HEAD_NUM			= 8
MODULES_NUM			= 6
ITERATIONS			= 4
OUT_SIZE			= 1 # binary as only left or right knee

# ##### Import Data #####

dataDirectory = '../../../AKOA_Analysis/'

def save_data():
	print("Saving Data...")

	print("Data Directory:", dataDirectory)

	# Need to sort data by patient so that we aren't leaking data between training and validation sets
	allPics = [dataDirectory + f for f in os.listdir(dataDirectory)]

	patients = [[0] * 2 for _ in range(len(allPics))]
	print("Number of Patients:", len(patients))

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
	print('Left Knees:', len(patients) - sum([i[1] for i in patients]))

	# Sort by substring
	patients = [list(i) for j, i in itertools.groupby(sorted(patients))]

	# TEMPORARY: REDUCE AMOUNT OF MEMORY USED!
	patients = patients[0:math.floor(len(patients) * 0.2)]

	# Split data
	print("Splitting data into training and testing sets")
	patients_train	= patients[0:math.floor(len(patients) * TEST_TRAINING_SPLIT)]
	patents_test	= patients[math.floor(len(patients) * TEST_TRAINING_SPLIT):-1]

	# Remove extra axis added by the above sorting line
	patients_train	= [item for sublist in patients_train for item in sublist]
	patients_test	= [item for sublist in patents_test for item in sublist]

	# Verify no leakage
	patients_train_array = [i[0] for i in patients_train]
	patients_train_array = np.array(patients_train_array)
	patients_test_array = [i[0] for i in patients_test]
	patients_test_array  = np.array(patients_test_array)
	print("Intersection: ", np.intersect1d(patients_train_array, patients_test_array))

	# Import/Load images
	print("Importing training images")
	xtrain = []
	ytrain = []
	for i in patients_train:
		for j in allPics:
			if i[0].split('.')[0] in j and i[0].split('.')[1] in j.split('de3')[1]:
				xtrain.append(np.asarray(PIL.Image.open(j).convert("L").resize(INPUT_SHAPE)))
				ytrain.append(i[1])
				break
	print("Importing testing images")
	xtest = []
	ytest = []
	for i in patients_test:
		for j in allPics:
			if i[0].split('.')[0] in j and i[0].split('.')[1] in j.split('de3')[1]:
				xtest.append(np.asarray(PIL.Image.open(j).convert("L").resize(INPUT_SHAPE)))
				ytest.append(i[1])
				break

	# Normalize the data to [0,1]
	print("Normalizing data")
	xtrain = np.array(xtrain, dtype=float)
	xtest  = np.array(xtest, dtype=float)
	xtrain[:] /= 255
	xtest[:] /= 255

	# Shuffle xtrain
	xtrain_order = list(range(0, len(xtrain)))
	random.shuffle(xtrain_order)
	xtrain = xtrain[random.shuffle(xtrain_order)]
	xtrain = np.squeeze(xtrain)

	# Shuffle xtest
	xtest_order = list(range(0, len(xtest)))
	random.shuffle(xtest_order)
	xtest = xtest[random.shuffle(xtest_order)]
	xtest = np.squeeze(xtest)

	# Save the data to local drive
	print("Saving data to disk")
	np.save('../../../xtrain', xtrain)
	np.save('../../../ytrain', ytrain)
	np.save('../../../xtest', xtest)
	np.save('../../../ytest', ytest)

# Save the data
if SAVE_DATA:
	save_data()

def pos_encoding(img):
	# Create grid
	n, x_size, y_size = img.shape
	img = tf.cast(img, tf.float32)
	x = tf.linspace(-1, 1, x_size)
	y = tf.linspace(-1, 1, y_size)
	xy_mesh = tf.meshgrid(x,y)
	xy_mesh = tf.transpose(xy_mesh)
	xy_mesh = tf.expand_dims(xy_mesh, -1)
	xy_mesh = tf.reshape(xy_mesh, [2,x_size,y_size,1])
	#print(xy_mesh)
	# Frequency logspace of nyquist f for bands
	up_lim = tf.math.log(MAX_FREQUENCY/2)/tf.math.log(10.)
	low_lim = math.log(1)
	f_sin = tf.math.sin(tf.experimental.numpy.logspace(low_lim, up_lim, BANDS) * math.pi)
	f_cos = tf.math.cos(tf.experimental.numpy.logspace(low_lim, up_lim, BANDS) * math.pi)
	# Add repeats
	xy_mesh = tf.repeat(xy_mesh, 2 * BANDS + 1, axis = 3)
	t = tf.concat([f_sin, f_cos], axis=0)
	t = tf.concat([t, [1.]], axis=0) # Size is now 2K+1
	# Get encoding/features
	encoding = xy_mesh * t
	encoding = tf.reshape(encoding, [1, x_size, y_size, (2 * BANDS + 1) * 2])
	encoding = tf.repeat(encoding, n, 0) # Repeat for all images (on first axis)
	img = tf.expand_dims(img, axis=3) # resize image data so that it fits
	out = tf.cast(encoding, tf.float32)
	out = tf.concat([img, out], axis=-1) # Add image data
	out = tf.reshape(out, [n, x_size * y_size, -1]) # Linearise
	return out

# Load Data
print("Loading Data")
xtrain = np.load('../../../xtrain.npy')
ytrain = np.load('../../../ytrain.npy')
xtest = np.load('../../../xtest.npy')
ytest = np.load('../../../ytest.npy')


'''
#xtrain = np.asarray(xtrain).astype('float32').reshape((-1,BATCH))
ytrain = np.asarray(ytrain).astype('float32').reshape((-1,BATCH))
#xtest = np.asarray(xtest).astype('float32').reshape((-1,BATCH))
ytest = np.asarray(ytest).astype('float32').reshape((-1,BATCH))
'''

print("xtrain shape:", xtrain.shape)
print("ytrain shape:", ytrain.shape)
print("xtest shape:", xtest.shape)
print("ytest shape:", ytest.shape)


#xtrain, ytrain, X_val, y_val, xtest, ytest = split_and_shuffle(xtrain, ytrain, xtest, ytest)

# Ensure data is batched properly
xtrain = xtrain[0:len(xtrain) // BATCH * BATCH]
ytrain = ytrain[0:len(ytrain) // BATCH * BATCH]
xtest = xtest[0:len(xtest) // BATCH * BATCH]
ytest = ytest[0:len(ytest) // BATCH * BATCH]

'''
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
train_dataset = train_dataset.batch(BATCH, drop_remainder = True)

test_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))
test_dataset = test_dataset.batch(BATCH, drop_remainder = True)
'''

# ##### Define Modules #####

def network_attention():
	# Network structure starting at latent array
	latent_layer = tf.keras.layers.Input(shape = [LATENT_ARRAY_SIZE, PROJECTION])

	#latent_layer = tf.keras.layers.LayerNormalization()(latent_layer) # Add a cheeky normalization layer
	query_layer  = tf.keras.layers.Dense(QKV_DIM)(latent_layer) # Query tensor (dense layer)

	# Network structure starting at byte array
	byte_layer  = tf.keras.layers.Input(shape = [BYTE_ARRAY_SIZE, PROJECTION])
	#byte_layer  = tf.keras.layers.LayerNormalization()(byte_layer) # Add a cheeky normalization layer
	key_layer   = tf.keras.layers.Dense(QKV_DIM)(byte_layer) # Key tensor (dense layer)
	value_layer = tf.keras.layers.Dense(QKV_DIM)(byte_layer) # Value tensor (dense layer)

	# Combine byte part into cross attention node thingy
	attention_layer = tf.keras.layers.Attention(use_scale=True)([query_layer, key_layer, value_layer])
	attention_layer = tf.keras.layers.Dense(QKV_DIM)(attention_layer)
	#attention_layer = tf.keras.layers.Dense(QKV_DIM)(attention_layer)
	attention_layer = tf.keras.layers.LayerNormalization()(attention_layer)

	# Combine latent array into cross attention node thingy
	attention_layer = tf.keras.layers.Add()([attention_layer, latent_layer]) # Add a connection straight from latent
	attention_layer = tf.keras.layers.LayerNormalization()(attention_layer)

	# Need to now add a connecting layer
	out = tf.keras.layers.Dense(PROJECTION, activation='relu')(attention_layer)
	out = tf.keras.layers.Dropout(DROPOUT_RATE)(out)
	#out = tf.keras.layers.Dense(PROJECTION)(out)

	attention_connect_layer = tf.keras.layers.Add()([out, attention_layer])

	out = tf.keras.Model(inputs = [latent_layer, byte_layer], outputs = attention_connect_layer)
	# Should probably also normalize
	return out

def network_transformer():
	# Get latent_size and PROJECTION
	latent_input_initial = tf.keras.layers.Input(shape = [LATENT_ARRAY_SIZE, PROJECTION])
	latent_input = copy.deepcopy(latent_input_initial)
	# Create as many transformer modules as necessary
	for i in range(TRANSFOMER_NUM):
		transformer_layer = tf.keras.layers.LayerNormalization()(latent_input_initial) # probs remove above normalization
		# Multihead attention layer
		transformer_layer = tf.keras.layers.MultiHeadAttention(num_heads = HEAD_NUM, key_dim = PROJECTION)(transformer_layer,transformer_layer)
		# Add dense layer
		transformer_layer = tf.keras.layers.Dense(PROJECTION)(transformer_layer)
		# Add passthrough connection from input
		transformer_layer = tf.keras.layers.Add()([latent_input_initial, transformer_layer])
		# Normalize for the fun of it
		transformer_layer = tf.keras.layers.LayerNormalization()(transformer_layer)
		# Get query
		x = tf.keras.layers.Dense(PROJECTION, activation='relu')(transformer_layer)
		x = tf.keras.layers.Dense(PROJECTION)(x)
		#x = tf.keras.layers.Dropout(DROPOUT_RATE)(x) # get some cheeky dropout
		# Add passthrough connection from transformer_layer
		transformer_layer = tf.keras.layers.Add()([x, transformer_layer])
		latent_input = transformer_layer # sketchy but also works
	# Add global pooling layer (not really part of transformer, but I don't care)
	out = tf.keras.Model(inputs = latent_input_initial, outputs = transformer_layer)
	return out

# ##### Create Perceiver Module #####

# Perceiver class
class Perceiver(tf.keras.Model):
	def __init__(self):
		super(Perceiver, self).__init__()

		#def build(self, input_shape):
		self.fourier_encoder = FourierEncode(10, 4)
		# TODO: Custom initializer to get truncated standard deviation thingy from paper
		self.in_layer = self.add_weight(shape = (LATENT_ARRAY_SIZE, PROJECTION), initializer = 'random_normal', trainable = True)
		#self.in_layer = tf.expand_dims(self.in_layer, axis = 0)
		# Add attention module
		self.attention = network_attention()
		# Add transformer module
		self.transformer = network_transformer()
		# Build
		#super(Perceiver, self).build(input_shape)

	def call(self, to_encode):
		# Attention input
		#frequency_data = pos_encoding(to_encode, BANDS, 10)
		frequency_data = self.fourier_encoder(to_encode)
		attention_in = [tf.expand_dims(self.in_layer, 0), frequency_data]
		#print(f_data.shape)
		#print(frequency_data)
		#attention_in = [self.in_layer, frequency_data]
		for _ in range(MODULES_NUM):
			#print("DDDDDDDDDDDDDDDDDDDDDDDDD")
			latent = self.attention(attention_in)
			#print("CCCCCCCCCCCCCCCCCCCCCCCCCCC")
			latent = self.transformer(latent)
			#print("BBBBBBBBBBBBBBBBBBBBBBBBBB")
			attention_in[0] = latent
			#print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
		out = tf.keras.layers.GlobalAveragePooling1D()(latent)
		final = tf.keras.layers.Dense(1, activation='sigmoid')(out)
		return final

# ##### Run Training/Evaluation #####

perceiver = Perceiver()

# Compile the model
perceiver.compile(
	optimizer = tfa.optimizers.LAMB(learning_rate=LEARNING_RATE, weight_decay_rate = 0.0002),
	loss = tf.keras.losses.BinaryCrossentropy(),
	metrics = tf.keras.metrics.BinaryAccuracy(name="accuracy")
)

model_history = perceiver.train_perceiver(train_set=(xtrain, ytrain), test_set=(xtest, ytest))

perceiver.summary()

