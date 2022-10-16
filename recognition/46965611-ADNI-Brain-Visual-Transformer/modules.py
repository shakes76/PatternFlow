"""
modules.py

Components of the visual transformer model.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers


class PatchLayer(Layer):
	"""
	Layer for transforming input images into patches.
	"""
	def __init__(self, patch_size):
		super(PatchLayer, self).__init__()
		self.patch_size = patch_size

	def call(self, images):
		batch_size = tf.shape(images)[0]
		patches = tf.image.extract_patches(
			images=images,
			sizes=[1, self.patch_size, self.patch_size, 1],
			strides=[1, self.patch_size, self.patch_size, 1],
			rates=[1, 1, 1, 1],
			padding='VALID'
		)
		patch_dims = patches.shape[-1]
		patches = tf.reshape(patches, [batch_size, -1, patch_dims])
		return patches


class FlattenAndEmbedPatch(Layer):
	"""
	Layer for projecting patches into a vector. Also adds a learnable
	position embedding to the projected vector.
	"""
	def __init__(self, num_patches, projection_dim):
		super(FlattenAndEmbedPatch, self).__init__()
		self.num_patches = num_patches
		self.projection_dim = projection_dim
		self.projection = layers.Dense(units=projection_dim)
		self.position_embedding = layers.Embedding(
			input_dim=self.num_patches, output_dim=self.projection_dim
		)
	
	def call(self, patch):
		positions = tf.range(0, self.num_patches, delta=1)
		return self.projection(patch) + self.position_embedding(positions)


def build_vision_transformer(input_shape, patch_size, num_patches,
		attention_heads, projection_dim, hidden_units, dropout_rate,
		transformer_layers):
	"""
	Builds the vision transformer model.
	"""
	# Input layer
	inputs = layers.Input(shape=input_shape)

	# Convert image data into patches
	patches = PatchLayer(patch_size)(inputs)
	
	# Encode patches
	encoded_patches = FlattenAndEmbedPatch(
		num_patches, projection_dim
	)(patches)

	# Create transformer layers
	for _ in range(transformer_layers):
		# First layer normalisation
		layer_norm_1 = layers.LayerNormalization(
			epsilon=1e-6
		)(encoded_patches)

		# Multi-head attention layer
		attention_output = layers.MultiHeadAttention(
			num_heads=attention_heads, key_dim=projection_dim,
			dropout=dropout_rate
		)(layer_norm_1, layer_norm_1)

		# First skip connection
		skip_1 = layers.Add()([attention_output, encoded_patches])

		# Second layer normalisation
		layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)(skip_1)

		# Multi-Layer Perceptron
		mlp_layer = layer_norm_2
		for units in hidden_units:
			mlp_layer = layers.Dense(units, activation=tf.nn.gelu)(mlp_layer)
			mlp_layer = layers.Dropout(dropout_rate)(mlp_layer)
		
		# Second skip connection
		encoded_patches = layers.Add()([mlp_layer, skip_1])

	# Create a [batch_size, projection_dim] tensor
	representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
	representation = layers.Flatten()(representation)
	representation = layers.Dropout(0.5)(representation)

	# MLP layer for learning features
	features = representation
	for units in [2048, 1024, 512]:
		features = layers.Dense(units, activation=tf.nn.gelu)(features)
		features = layers.Dropout(0.5)(features)

	# Classify outputs
	logits = layers.Dense(2)(features)

	# Create Keras model
	model = tf.keras.Model(inputs=inputs, outputs=logits)

	return model