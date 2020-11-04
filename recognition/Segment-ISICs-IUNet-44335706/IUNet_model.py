'''
Module for building an improved UNet model using Keras layers
'''

import tensorflow as tf

def iunet_model():
	inputs = tf.keras.Input((256, 256, 1))

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

	# Upsampling phase
	depths = (128, 64, 32)
	for i in range(3):
		# First convolutional layer
		layer = tf.keras.layers.Conv2DTranspose(depths[i], 3, padding="same", activation="relu")(layer)
		layer = tf.keras.layers.BatchNormalization()(layer)
		
		# Add dropout in between blocks to prevent overfitting
		layer = tf.keras.layers.Dropout(0.2)(layer)

		# Second convolutional layer
		layer = tf.keras.layers.Conv2DTranspose(depths[i], 3, padding="same", activation="relu")(layer)
		layer = tf.keras.layers.BatchNormalization()(layer)
		
		# Normalise the batch and upsample
		
		layer = tf.keras.layers.UpSampling2D(2)(layer)
		layer = tf.keras.layers.Concatenate(axis=3)([layer, layers[-i-1]])

	# Classify at the final output layer with softmax
	outputs = tf.keras.layers.Conv2D(2, 3, activation="softmax", padding="same")(layer)

	# Define the model
	return tf.keras.Model(inputs, outputs)
