"""
VQVAE components for COMP3710 VQVAE project

Inspired by https://keras.io/examples/generative/vq_vae/
"""

# Name collisions in imports :((( have to spell out exact model paths and clog up the code

#import	tensorflow_probability	as		tfp
import	tensorflow				as		tf
from	tensorflow				import	keras
from	tensorflow.keras		import	layers
#rom	dataset					import	ENC_IN_SHAPE

FLATTEN				= -1
STRIDES				= 2
PCNN_STRIDES		= 1
KERN_SIZE			= 3
CONV1_KERN_SIZE		= 1
CONV_W_FACTOR		= 32
FILTER_FACTOR		= 2
KERN_FACTOR			= 2
NO_FILTERS			= 128
PCNN_IN_KERN_SIZE	= 7
PCNN_OUT_KERN_SIZE	= 1
PCNN_MID_KERN_SIZE	= 1
KERN_INIT			= 1.0
RESIDUAL_BLOCKS		= 2
ENC_IN_SHAPE		= (80, 80, 1)

class VectorQuantiser(layers.Layer):
	"""
	Custom layer to handle vector quantisation
	"""
	def __init__(self, n_embeds, embed_dim, beta, **kwargs):
		super().__init__(**kwargs)
		self.embed_dim		= embed_dim
		self.n_embeds		= n_embeds
		self.beta			= (beta)
		w_init				= tf.random_uniform_initializer()
		self.embeddings		= tf.Variable(
			initial_value	= w_init(
				shape = (self.embed_dim, self.n_embeds), dtype = "float32"
			), trainable = True, name = "embeddings_vqvae")

	def code_indices(self, flat):
		sim		= tf.matmul(flat, self.embeddings)
		dists	= tf.reduce_sum(flat ** 2, axis = 1, keepdims = True) + tf.reduce_sum(self.embeddings ** 2, axis = 0) - 2 * sim
		encodes	= tf.argmin(dists, axis = 1)
		return encodes

	def call(self, x):
		"""
		Forward computation handler

		x		- inputs from previous layer

		return	- vector quantisation of inputs from previous layer
		"""

		in_shape	= tf.shape(x)
		flat		= tf.reshape(x, [FLATTEN, self.embed_dim])
		enc_ind		= self.code_indices(flat)
		enc			= tf.one_hot(enc_ind, self.n_embeds)
		qtised		= tf.matmul(enc, self.embeddings, transpose_b = True)
		qtised		= tf.reshape(qtised, in_shape)
		commit		= self.beta * tf.reduce_mean(tf.stop_gradient(qtised) - x) ** 2
		codebook	= tf.reduce_mean((qtised - tf.stop_gradient(x)) ** 2)
		self.add_loss(commit + codebook)
		qtised		= x + tf.stop_gradient(qtised - x)

		return qtised

	def get_config(self):
		"""So that saving actually works"""
		config = super().get_config()
		config.update({
			"n_embeds"	: self.n_embeds,
			"embed_dim"	: self.embed_dim,
			"beta"		: self.beta,
		})

		return config

def encoder(lat_dim):
	"""
	Builds the encoder-half of the VQVAE

	lat_dim	- the dimensionality of the latent space

	return	- the encoder half of the VQVAE
	"""

	enc_in	= keras.Input(shape = ENC_IN_SHAPE)
	convos	= layers.Conv2D(CONV_W_FACTOR,		KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(enc_in)
	convos2	= layers.Conv2D(2 * CONV_W_FACTOR,	KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(convos)
	enc_out	= layers.Conv2D(lat_dim, 1, padding = "same")(convos2)

	return keras.Model(enc_in, enc_out, name = "encoder")


def decoder(lat_dim):
	"""
	Builds the decoder-half of the VQVAE

	lat_dim	- the dimensionality of the latent space

	return	- the decoder half of the VQVAE
	"""

	lat_in	= keras.Input(shape = encoder(lat_dim).output.shape[1:])
	convos	= layers.Conv2DTranspose(2 * CONV_W_FACTOR,	KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(lat_in)
	convos2	= layers.Conv2DTranspose(CONV_W_FACTOR,		KERN_SIZE, activation = "relu", strides = STRIDES, padding = "same")(convos)
	dec_out	= layers.Conv2DTranspose(1, KERN_SIZE, padding = "same")(convos2)


	return keras.Model(lat_in, dec_out, name = "decoder")

def build_vqvae(lat_dim, embeds, beta):
	"""
	Sandwich together the encoder + VQ + decoder

	lat_dim	- dimensionality of the latent space
	embeds	- number of embeddings in the VQ
	beta	- beta param for the VQ

	return	- the assembled VQVAE
	"""

	vq			= VectorQuantiser(embeds, lat_dim, beta, name = "vector_quantiser")
	enc			= encoder(lat_dim)
	dec			= decoder(lat_dim)
	ins			= keras.Input(shape = ENC_IN_SHAPE)
	enc_out		= enc(ins)
	qtised		= vq(enc_out)
	reconstruct	= dec(qtised)

	return keras.Model(ins, reconstruct, name = "vqvae")

class Trainer(keras.models.Model):
	"""Wrapper class for vqvae training functionality"""
	def __init__(self, train_vnce, lat_dim, n_embeds, beta, **kwargs):
		super(Trainer, self).__init__(**kwargs)
		self.train_vnce	= train_vnce
		self.lat_dim	= lat_dim
		self.n_embeds	= n_embeds
		self.vqvae		= build_vqvae(self.lat_dim, self.n_embeds, beta)
		self.tot_loss	= keras.metrics.Mean()
		self.recon_loss	= keras.metrics.Mean()
		self.vq_loss	= keras.metrics.Mean()

	@property
	def metrics(self):
		"""Total loss, reconstruction loss, vector quantisation loss"""
		return [self.tot_loss, self.recon_loss, self.vq_loss]

	def train_step(self, x):
		"""Individual VQVAE training step"""
		with tf.GradientTape() as tape:
			# Losses
			recons		= self.vqvae(x)
			recon_loss	= (tf.reduce_mean((x - recons) ** 2) / self.train_vnce)
			total_loss	= recon_loss + sum(self.vqvae.losses)

		# Backpropagation
		grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
		self.tot_loss.update_state(total_loss)
		self.recon_loss.update_state(recon_loss)
		self.vq_loss.update_state(sum(self.vqvae.losses))

		# Results
		return {
			"loss"					:	self.tot_loss.result(),
			"reconstruction_loss"	:	self.recon_loss.result(),
			"vqvae_loss"			:	self.vq_loss.result()
		}

class PixelConvolution(layers.Layer):
	"""Pixel convolutional layer for pixel cnn"""
	def __init__(self, mask, **kwargs):
		super().__init__()
		self.mask = mask
		self.conv = Conv2D(**kwargs)

	def build(self, input_shape):
		"""Construct the convolutional kernel"""
		self.conv.build(input_shape)
		kern_shape	= self.conv.kernel.get_shape()
		self.mask	= zeros(shape = kern_shape)
		self.mask[: kern_shape[0] // KERN_FACTOR, ...] =KERN_INIT
		self.mask[kern_shape[0] // KERN_FACTOR, : kern_shape[1] // KERN_FACTOR, ...] = KERN_INIT
		if self.mask == "B":
			self.mask[kernel_shape[0] // KERN_FACTOR, kern_shape[1] // KERN_FACTOR, ...] = KERN_INIT

	def call(self, inputs):
		"""Forward computational handler"""
		self.conv.kernel.assign(self.conv.kernel * self.mask)
		return self.conv(inputs)

class ResidualBlock(layers.Layer):
	"""Resnet block based on PixelConvolution layers"""
	def __init__(self, filters, **kwargs):
		super(ResidualBlock, self).__init__(**kwargs)
		self.conv1 = Conv2D(
			filters = filters, kernel_size = CONV1_KERN_SIZE, activation="relu"
		)
		self.pixel_conv = PixelConvolution(
			mask		= "B",
			filters		= filters // FILTER_FACTOR,
			kernel_size	= KERN_SIZE,
			activation	= "relu",
			padding		= "same",
		)
		self.conv2 = Conv2D(filters = filters, kernel_size = CONV1_KERN_SIZE, activation = "relu")

	def call(self, inputs):
		"""Forward computation handler"""
		layer	= self.conv1(inputs)
		layer	= self.pixel_conv(layer)
		layer	= self.conv2(layer)

		return add([inputs, layer])

def build_pcnn(in_shape, no_resid_blocks, no_pcnn_layers, trainer):
	"""Construct the Pixel CNN

	in_shape		- shape of PCNN inputs
	no_resid_blocks	- the number of residual blocks within the PCNN
	no_pcnn_layers	- the number of layers within the PCNN
	trainer			- the training wrapper class for the VQVAE

	return			- the assembled Pixel CNN
	"""

	pcnn_ins	= Input(shape = in_shape, dtype = tf.int32)
	onehot		= tf.one_hot(pcnn_ins, trainer.n_embeds)
	layer		= PixelConvolution(mask = "A", filters = NO_FILTERS, kernel_size = PCNN_IN_KERN_SIZE, activation = "relu", padding = "same")(onehot)

	for _ in range(no_resid_blocks):
		layer = ResidualBlock(filters = NO_FILTERS)(layer)

	for _ in range(no_pcnn_layers):
		layer = PixelConvolution(
			mask		= "B",
			filters		= NO_FILTERS,
			kernel_size	= PCNN_MID_KERN_SIZE,
			strides		= PCNN_STRIDES,
			activation	= "relu",
			padding		= "valid",
		)(layer)

	out		= Conv2D(filters = trainer.n_embeds, kernel_size = PCNN_OUT_KERN_SIZE, strides = PCNN_STRIDES, padding = "valid")(layer)
	pcnn	= Model(pcnn_ins, out)
	pcnn.summary()

	return pcnn
