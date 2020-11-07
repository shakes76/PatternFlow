'''
Module for building an improved UNet model using Keras layers
'''

import tensorflow as tf


def dice_coef_loss(y_act, y_pred):
	'''
	Return the dice coefficient loss of two images
	'''
	flat_act = tf.cast(tf.keras.backend.flatten(y_act), dtype=tf.float32)
	flat_pred = tf.cast(tf.keras.backend.flatten(y_pred), dtype=tf.float32)
	intersection = tf.reduce_sum(tf.multiply(flat_act, flat_pred))
	return 1 - 2 * (intersection + 1) / (tf.reduce_sum(flat_act) + tf.reduce_sum(flat_pred) + 1)


def iunet_model():
	'''
	Function for building an iunet Keras model
	'''
	inputs = tf.keras.Input((256, 256, 3))

	# Start block
	layer = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
	layer = tf.keras.layers.BatchNormalization()(layer)

	# List for storing layers for concatenating
	layers = []

	# Create model
	# Downsampling phase
	for depth in (64, 128, 256):
		# First convolutional layer
		layer = tf.keras.layers.Conv2D(depth, 3, padding="same", activation="relu")(layer)
		layer = tf.keras.layers.BatchNormalization()(layer)
		
		# Add dropout in between blocks to prevent overfitting
		layer = tf.keras.layers.Dropout(0.2)(layer)

		# Second convolutional layer
		layer = tf.keras.layers.Conv2D(depth, 3, padding="same", activation = "relu")(layer)
		layer = tf.keras.layers.BatchNormalization()(layer)
		
		# Store layer in list
		layers.append(layer)
		
		# Pool down
		layer = tf.keras.layers.MaxPooling2D((2, 2))(layer)
	
	# Central block
	# First convolutional layer
	layer = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	
	# Add dropout in between blocks to prevent overfitting
	layer = tf.keras.layers.Dropout(0.2)(layer)

	# First convolutional layer
	layer = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)

	# Upsampling phase
	depths = (256, 128, 64)
	for i in range(3):

		# First convolutional layer
		layer = tf.keras.layers.Conv2DTranspose(depths[i], 3, padding="same", activation="relu")(layer)
		layer = tf.keras.layers.BatchNormalization()(layer)
		
		# Add dropout in between blocks to prevent overfitting
		layer = tf.keras.layers.Dropout(0.2)(layer)

		# Second convolutional layer
		layer = tf.keras.layers.Conv2DTranspose(depths[i], 3, padding="same", activation="relu")(layer)
		layer = tf.keras.layers.BatchNormalization()(layer)

		# Upsample and concatenate layers
		layer = tf.keras.layers.UpSampling2D(2)(layer)
		layer = tf.keras.layers.Concatenate(axis=3)([layer, layers[-i-1]])

	# Classify at the final output layer with softmax
	outputs = tf.keras.layers.Conv2D(2, 3, activation="softmax", padding="same")(layer)

	# Define the model
	return tf.keras.Model(inputs, outputs)

