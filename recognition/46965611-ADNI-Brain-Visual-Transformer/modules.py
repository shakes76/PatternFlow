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
		self.num_patches = num_patches
		self.projection_dim = projection_dim
		self.projection = layers.Dense(units=projection_dim)
		self.position_embedding = layers.Embedding(
			input_dim=self.num_patches, output_dim=projection_dim
		)
	
	def call(self, patch):
		positions = tf.range(0, self.num_patches)
		return self.projection(patch) + self.position_embedding(positions)

class MLP(Layer):
	"""
	Multi-layer perceptron to be used in the transformer encoder.
	"""
	def __init__(self, previous_layer, hidden_units, dropout_rate):
		self.x = previous_layer
		self.hidden_units = hidden_units
		self.dropout = dropout_rate

	def call(self):
		for units in self.hidden_units:
			x = layers.Dense(units, activation=tf.nn.gelu)(x)
			x = layers.Dropout(self.dropout)(x)
		return x