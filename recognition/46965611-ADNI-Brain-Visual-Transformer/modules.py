"""
modules.py

Components of the visual transformer model.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import math


class PatchLayer(Layer):
	"""
	Layer for shifting inputted images and transforming images into patches.
	"""
	def __init__(self, image_size, patch_size, num_patches, projection_dim):
		super(PatchLayer, self).__init__()
		self.image_size = image_size
		self.patch_size = patch_size
		self.half_patch = patch_size // 2
		self.flatten_patches = layers.Reshape((num_patches, -1))
		self.projection = layers.Dense(units=projection_dim)
		self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

	def shift_images(self, images, mode):
		# Build diagonally shifted images
		if mode == 'left-up':
			crop_height = self.half_patch
			crop_width = self.half_patch
			shift_height = 0
			shift_width = 0
		elif mode == 'left-down':
			crop_height = 0
			crop_width = self.half_patch
			shift_height = self.half_patch
			shift_width = 0
		elif mode == 'right-up':
			crop_height = self.half_patch
			crop_width = 0
			shift_height = 0
			shift_width = self.half_patch
		else:
			crop_height = 0
			crop_width = 0
			shift_height = self.half_patch
			shift_width = self.half_patch

		crop = tf.image.crop_to_bounding_box(
			images,
			offset_height=crop_height,
			offset_width=crop_width,
			target_height=self.image_size - self.half_patch,
			target_width=self.image_size - self.half_patch
		)

		shift_pad = tf.image.pad_to_bounding_box(
			crop,
			offset_height=shift_height,
			offset_width=shift_width,
			target_height=self.image_size,
			target_width=self.image_size
		)
		return shift_pad

	def call(self, images):
		images = tf.concat(
			[
				images,
				self.shift_images(images, mode='left-up'),
				self.shift_images(images, mode='left-down'),
				self.shift_images(images, mode='right-up'),
				self.shift_images(images, mode='right-down'),
			],
			axis=-1
		)
		patches = tf.image.extract_patches(
			images=images,
			sizes=[1, self.patch_size, self.patch_size, 1],
			strides=[1, self.patch_size, self.patch_size, 1],
			rates=[1, 1, 1, 1],
			padding='VALID'
		)
		flat_patches = self.flatten_patches(patches)
		tokens = self.layer_norm(flat_patches)
		tokens = self.projection(tokens)

		return (tokens, patches)


class EmbedPatch(Layer):
	"""
	Layer for projecting patches into a vector. Also adds a learnable
	position embedding to the projected vector.
	"""
	def __init__(self, num_patches, projection_dim):
		super(EmbedPatch, self).__init__()
		self.num_patches = num_patches
		self.position_embedding = layers.Embedding(
			input_dim=self.num_patches, output_dim=projection_dim
		)
	
	def call(self, patches):
		positions = tf.range(0, self.num_patches, delta=1)
		return patches + self.position_embedding(positions)


class MultiHeadAttentionLSA(layers.MultiHeadAttention):
	def __init__(self, **kwargs):
		super(MultiHeadAttentionLSA, self).__init__(**kwargs)
		self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

	def _compute_attention(self, query, key, value, attention_mask=None,
			training=None):
		query = tf.multiply(query, 1.0/self.tau)
		attention_scores = tf.einsum(self._dot_product_equation, key, query)
		attention_scores = self._masked_softmax(attention_scores, attention_mask)
		attention_scores_dropout = self._dropout_layer(
			attention_scores, training=training
		)
		attention_output = tf.einsum(
			self._combine_equation, attention_scores_dropout, value
		)
		return attention_output, attention_scores

def build_vision_transformer(input_shape, image_size, patch_size, num_patches,
			attention_heads, projection_dim, hidden_units, dropout_rate,
			transformer_layers):
	"""
	Builds the vision transformer model.
	"""
	# Input layer
	inputs = layers.Input(shape=input_shape)

	# Convert image data into patches
	(tokens, _) = PatchLayer(
		image_size,
		patch_size,
		num_patches,
		projection_dim
	)(inputs)
	
	# Encode patches
	encoded_patches = EmbedPatch(num_patches, projection_dim)(tokens)

	# Create transformer layers
	for _ in range(transformer_layers):
		# First layer normalisation
		layer_norm_1 = layers.LayerNormalization(
			epsilon=1e-6
		)(encoded_patches)

		# Build diagoanl attention mask
		diag_attn_mask = 1 - tf.eye(num_patches)
		diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

		# Multi-head attention layer
		attention_output = MultiHeadAttentionLSA(
			num_heads=attention_heads, key_dim=projection_dim,
			dropout=dropout_rate
		)(layer_norm_1, layer_norm_1, attention_mask=diag_attn_mask)

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
	for units in [2048, 1024]:
		features = layers.Dense(units, activation=tf.nn.gelu)(features)
		features = layers.Dropout(0.5)(features)

	# Classify outputs
	logits = layers.Dense(2)(features)

	# Create Keras model
	model = tf.keras.Model(inputs=inputs, outputs=logits)

	return model