# General References:
# - https://keras.io/examples/vision/perceiver_image_classification/ (good resource to see general structure)
# - https://github.com/Rishit-dagli/Perceiver (decent for seeing how some functions are implemented)
# - https://keras.io/api/layers/base_layer/ (how to use classes with tf)

# Import parameters
import Parameters

# Libraries needed for model
import tensorflow as tf
import tensorflow.keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import copy
import math

print("Tensorflow Version:", tf.__version__)
tf.config.run_functions_eagerly(True)

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
	up_lim = tf.math.log(Parameters.MAX_FREQUENCY/2)/tf.math.log(10.)
	low_lim = math.log(1)
	f_sin = tf.math.sin(tf.experimental.numpy.logspace(low_lim, up_lim, Parameters.BANDS) * math.pi)
	f_cos = tf.math.cos(tf.experimental.numpy.logspace(low_lim, up_lim, Parameters.BANDS) * math.pi)
	# Add repeats
	xy_mesh = tf.repeat(xy_mesh, 2 * Parameters.BANDS + 1, axis = 3)
	t = tf.concat([f_sin, f_cos], axis=0)
	t = tf.concat([t, [1.]], axis=0) # Size is now 2K+1
	# Get encoding/features
	encoding = xy_mesh * t
	encoding = tf.reshape(encoding, [1, x_size, y_size, (2 * Parameters.BANDS + 1) * 2])
	encoding = tf.repeat(encoding, n, 0) # Repeat for all images (on first axis)
	img = tf.expand_dims(img, axis=3) # resize image data so that it fits
	out = tf.cast(encoding, tf.float32)
	out = tf.concat([img, out], axis=-1) # Add image data
	out = tf.reshape(out, [n, x_size * y_size, -1]) # Linearise
	return out

# ##### Define Modules #####

def network_attention():
	# Network structure starting at latent array
	latent_layer = tf.keras.layers.Input(shape = [Parameters.LATENT_ARRAY_SIZE, Parameters.PROJECTION])

	#latent_layer = tf.keras.layers.LayerNormalization()(latent_layer) # Add a cheeky normalization layer
	query_layer  = tf.keras.layers.Dense(Parameters.QKV_DIM)(latent_layer) # Query tensor (dense layer)

	# Network structure starting at byte array
	byte_layer  = tf.keras.layers.Input(shape = [Parameters.BYTE_ARRAY_SIZE, Parameters.PROJECTION])
	#byte_layer  = tf.keras.layers.LayerNormalization()(byte_layer) # Add a cheeky normalization layer
	key_layer   = tf.keras.layers.Dense(Parameters.QKV_DIM)(byte_layer) # Key tensor (dense layer)
	value_layer = tf.keras.layers.Dense(Parameters.QKV_DIM)(byte_layer) # Value tensor (dense layer)

	# Combine byte part into cross attention node thingy
	attention_layer = tf.keras.layers.Attention(use_scale=True)([query_layer, key_layer, value_layer])
	attention_layer = tf.keras.layers.Dense(Parameters.QKV_DIM)(attention_layer)
	#attention_layer = tf.keras.layers.Dense(QKV_DIM)(attention_layer)
	attention_layer = tf.keras.layers.LayerNormalization()(attention_layer)

	# Combine latent array into cross attention node thingy
	attention_layer = tf.keras.layers.Add()([attention_layer, latent_layer]) # Add a connection straight from latent
	attention_layer = tf.keras.layers.LayerNormalization()(attention_layer)
	
	# Need to now add a connecting layer
	out = tf.keras.layers.Dense(Parameters.PROJECTION, activation='relu')(attention_layer)
	out = tf.keras.layers.Dropout(Parameters.DROPOUT_RATE)(out)
	#out = tf.keras.layers.Dense(PROJECTION)(out)
	
	attention_connect_layer = tf.keras.layers.Add()([out, attention_layer])

	out = tf.keras.Model(inputs = [latent_layer, byte_layer], outputs = attention_connect_layer)
	# Should probably also normalize
	return out

def network_transformer():
	# Get latent_size and PROJECTION
	latent_input_initial = tf.keras.layers.Input(shape = [Parameters.LATENT_ARRAY_SIZE, Parameters.PROJECTION])
	latent_input = copy.deepcopy(latent_input_initial)
	# Create as many transformer modules as necessary
	for i in range(Parameters.TRANSFOMER_NUM):
		transformer_layer = tf.keras.layers.LayerNormalization()(latent_input_initial) # probs remove above normalization
		# Multihead attention layer
		transformer_layer = tf.keras.layers.MultiHeadAttention(num_heads = Parameters.HEAD_NUM, key_dim = Parameters.PROJECTION)(transformer_layer,transformer_layer)
		# Add dense layer
		transformer_layer = tf.keras.layers.Dense(Parameters.PROJECTION)(transformer_layer)
		# Add passthrough connection from input
		transformer_layer = tf.keras.layers.Add()([latent_input_initial, transformer_layer])
		# Normalize for the fun of it
		transformer_layer = tf.keras.layers.LayerNormalization()(transformer_layer)
		# Get query
		x = tf.keras.layers.Dense(Parameters.PROJECTION, activation='relu')(transformer_layer)
		x = tf.keras.layers.Dense(Parameters.PROJECTION)(x)
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
		# TODO: Custom initializer to get truncated standard deviation thingy from paper
		self.in_layer = self.add_weight(shape = (Parameters.LATENT_ARRAY_SIZE, Parameters.PROJECTION), initializer = 'random_normal', trainable = True)
		self.in_layer = tf.expand_dims(self.in_layer, axis = 0)
		# Add attention module
		self.attention = network_attention()
		# Add transformer module
		self.transformer = network_transformer()
		# Define classification layer
		self.classification = tf.keras.layers.Dense(units=1, activation='sigmoid')

	def call(self, to_encode):
		# Attention input
		frequency_data = pos_encoding(to_encode)
		attention_in = [self.in_layer, frequency_data]
		# Loop for repeats
		for _ in range(Parameters.ITERATIONS):
			# Add cross-attemtion layer
			latent = self.attention(attention_in)
			# Add transformer layer
			latent = self.transformer(latent)
			attention_in[0] = latent
		# Pool layers together
		out = tf.keras.layers.GlobalAveragePooling1D()(latent)
		final = self.classification(out)
		return final

	def train_perceiver(self, xtrain, ytrain, xtest, ytest):

		# Ensure proper batching of data
		xtrain = xtrain[0:(len(xtrain) // Parameters.BATCH) * Parameters.BATCH]
		ytrain = ytrain[0:(len(ytrain) // Parameters.BATCH) * Parameters.BATCH]
		xtest = xtest[0:(len(xtest) // Parameters.BATCH) * Parameters.BATCH]
		ytest = ytest[0:(len(ytest) // Parameters.BATCH) * Parameters.BATCH]

		self.compile(
			optimizer=tfa.optimizers.LAMB(learning_rate=Parameters.LEARNING_RATE, weight_decay_rate=Parameters.WEIGHT_DECAY_RATE),
			loss=tf.keras.losses.BinaryCrossentropy(),
			metrics=[tf.keras.metrics.BinaryAccuracy()]
		)

		model_history = self.fit(
			xtrain, ytrain,
			epochs = Parameters.EPOCHS,
			batch_size = Parameters.BATCH,
			validation_split = Parameters.VALIDATION_SPLIT,
			validation_batch_size = Parameters.BATCH
		)

		_, accuracy = self.evaluate(xtest, ytest)

		return [model_history, accuracy]
