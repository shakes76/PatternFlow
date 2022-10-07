"""
VQVAE components for COMP3710 VQVAE project

Inspired by https://keras.io/examples/generative/vq_vae/
"""

import	tensorflow				as		tf
from	tensorflow.keras		import	Input,	Model
from	tensorflow.keras.layers	import	Conv2D,	Conv2DTranspose, Layer

FLATTEN			= -1
ENC_IN_SHAPE	= (28, 28, 1)
STRIDES			= 2
KERN_SIZE		= 3
N_FIRST			= 1
CONV_W_FACTOR	= 32

class VectorQuantizer(Layer):
	"""
	Custom layer to handle vector quantisation
	"""
	def __init__(self, embeds, embed_dim, beta, **kwargs):
		super().__init__(**kwargs)
		self.embed_dim	= embed_dim
		self.embeds		= embeds
		self.beta		= beta
		self.embeddings = tf.Variable(
			initial_value = tf.random_uniform_initializer()(
				shape = (self.embed_dim, self.embeds), dtype="float32"
			),
			trainable = True
		)

	def call(self, x):
		"""
		Forward computation handler

		x		- inputs from previous layer

		return	- vector quantisation of inputs from previous layer
		"""

		in_shape	= tf.shape(x)
		flat		= tf.reshape(x, [FLATTEN, self.embed_dim])
		dists		= (tf.reduce_sum(flat ** 2, axis = 1, keepdims = True)
						+ tf.reduce_sum(self.embeddings ** 2, axis = 0)
						- 2 * tf.matmul(flat, self.embeddings))
		enc_ind		= tf.argmin(dists, axis = 1)
		enc			= tf.one_hot(enc_ind, self.embeds)
		qtised		= tf.matmul(enc, self.embeddings, transpose_b=True)
		qtised		= tf.reshape(qtised, in_shape)
		commit		= tf.reduce_mean((tf.stop_gradient(qtised) - x) ** 2)
		codebook	= tf.reduce_mean((qtised - tf.stop_gradient(x)) ** 2)
		self.add_loss(self.beta * commit + codebook)
		qtised		= x + tf.stop_gradient(qtised - x)

		return qtised

def encoder(lat_dim):
	"""
	Builds the encoder-half of the VQVAE

	lat_dim	- the dimensionality of the latent space

	return	- the encoder half of the VQVAE
	"""

	enc_in	= Input(shape = ENC_IN_SHAPE)
	x		= Conv2D(CONV_W_FACTOR,		KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(enc_in)
	x		= Conv2D(2 * CONV_W_FACTOR,	KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(x)
	enc_out	= Conv2D(lat_dim, 1, padding = "same")(x)

	return Model(enc_in, enc_out)

def decoder(lat_dim):
	"""
	Builds the decoder-half of the VQVAE

	lat_dim	- the dimensionality of the latent space

	return	- the decoder half of the VQVAE
	"""

	lat_in	= Input(shape = encoder(lat_dim).output.shape[N_FIRST:])
	x		= Conv2DTranspose(2 * CONV_W_FACTOR,	KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(lat_in)
	x		= Conv2DTranspose(CONV_W_FACTOR,		KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(x)
	dec_out	= Conv2DTranspose(1, KERN_SIZE, padding = "same")(x)

	return Model(lat_in, dec_out)

def build(lat_dim, embeds, beta):
	"""
	Sandwich together the encoder + VQ + decoder

	lat_dim	- dimensionality of the latent space
	embeds	- number of embeddings in the VQ
	beta	- beta param for the VQ

	return	- the assembled VQVAE
	"""

	vq			= VectorQuantizer(embeds, lat_dim, beta)#, name="vector_quantizer")
	enc			= encoder(lat_dim)
	dec			= decoder(lat_dim)
	ins			= Input(shape = ENC_IN_SHAPE)
	enc_out		= enc(ins)
	qtised		= vq(enc_out)
	reconstruct	= dec(qtised)

	return Model(ins, reconstruct)
