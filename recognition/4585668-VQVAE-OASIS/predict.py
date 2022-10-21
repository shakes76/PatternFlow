"""
Load VQVAE & PCNN models and evaluate performance
"""

import	tensorflow				as		tf
import	tensorflow_probability	as		tfp
from	numpy.random			import	choice
from	tensorflow.image		import	convert_image_dtype, ssim
from	matplotlib.pyplot		import	suptitle, imshow, subplot, axis, show, cm, title
from	keras.models			import	load_model
from	numpy					import	var, zeros
from	dataset					import	get_ttv, normalise
from	modules					import	VectorQuantiser, PixelConvolution, ResidualBlock, Trainer, FLAT
from	train					import	PCNN_PATH, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA, OPTIMISER, MODEL_PATH

VISUALISATIONS	= 5
COLS			= 2
MAX_VAL			= 1.0
CENTRE			= 0.5
GREY			= cm.gray
BATCH			= 10

def compare(originals, recons):
	"""Compare original test brains to encoded/decoded reconstruction images

	originals	- the original brain images
	recons		- original images after encoding and decoding

	return		- average SSIM between the original images and their reconstructions
	"""

	ssims	= 0
	pairs	= zip(originals, recons)
	for i, pair in enumerate(pairs):
		o, r	= pair
		orig	= convert_image_dtype(o, tf.float32)
		recon	= convert_image_dtype(r, tf.float32)
		sim		= ssim(orig, recon, max_val = MAX_VAL)
		ssims	+= sim
		subplot(VISUALISATIONS, COLS, COLS * i + 1)
		imshow(o + CENTRE, cmap = GREY)
		title("Test Input")
		axis("off")
		subplot(VISUALISATIONS, COLS, COLS * (i + 1))
		imshow(r + CENTRE, cmap = GREY)
		title("Test Reconstruction")
		axis("off")
	suptitle("SSIM: %.2f" %(ssims / len(originals)))
	show()

	return ssims

def validate_vqvae(vqvae, test):
	"""Evaluate how well the VQVAE performed using a test set

	vqvae	- the trained VQVAE
	test	- set of test images for evaluating the VQVAE
	"""

	image_inds	= choice(len(test), VISUALISATIONS)
	images		= test[image_inds]
	recons		= vqvae.predict(images)
	ssim		= compare(images, recons)
	avg_ssim	= ssim / VISUALISATIONS
	print(f"Average SSIM: {avg_ssim}")

def show_new_brains(priors, samples):
	"""Display encoded priors to their respective decoded images

	priors	- set of encodings from the PCNN
	samples	- images obtained by decoding priors with the VQVAE's decoder
	"""

	for i in range(VISUALISATIONS):
		subplot(VISUALISATIONS, COLS, COLS * i + 1)
		imshow(priors[i], cmap = GREY)
		title("PCNN Prior")
		axis("off")
		subplot(VISUALISATIONS, COLS, COLS * (i + 1))
		imshow(samples[i] + CENTRE, cmap = GREY)
		title("Decoded Prior")
		axis("off")
	show()

def show_quantisations(test, encodings, quantiser):
	"""Display original images to their respective encodings

	test		- images to be encoded
	encodings	- encodings obtained from the PCNN
	quantiser	- vector quantiser from VQVAE
	"""

	encodings	= encodings[:len(encodings) // 2] # Throw out half the encodings because I have no memory
	flat		= encodings.reshape(FLAT, encodings.shape[FLAT])
	codebooks	= quantiser.code_indices(flat).numpy().reshape(encodings.shape[:FLAT])

	for i in range(VISUALISATIONS):
		subplot(VISUALISATIONS, COLS, COLS * i + 1)
		imshow(test[i] + CENTRE, cmap = GREY)
		title("Test Image")
		axis("off")
		subplot(VISUALISATIONS, COLS, COLS * (i + 1))
		imshow(codebooks[i], cmap = GREY)
		title("VQ Encoding")
		axis("off")
	show()

def validate_pcnn(vqvae, train_vnce, test):
	"""Evaluate the performance of the PCNN

	vqvae		- VQVAE model
	train_vnce	- variance of the training set post-normalisation
	test		- set of images used to evaluate the PCNN
	"""

	pcnn	= load_model(PCNN_PATH, custom_objects = {"PixelConvolution": PixelConvolution, "ResidualBlock": ResidualBlock})
	priors	= zeros(shape = (BATCH,) + (pcnn.input_shape)[1:])
	batch, rows, columns = priors.shape

	for r in range(rows):
		for c in range(columns):
			logs	= pcnn.predict(priors)
			sampler	= tfp.distributions.Categorical(logs)
			prob	= sampler.sample()
			priors[:, r, c] = prob[:, r, c]


	encoder		= vqvae.get_layer("encoder")
	quantiser	= vqvae.get_layer("quantiser")
	encoded_out	= encoder.predict(test)
	show_quantisations(test, encoded_out, quantiser)
	old_embeds	= quantiser.embeddings
	pr_onehots	= tf.one_hot(priors.astype("int32"), NUM_EMBEDDINGS).numpy()
	qtised		= tf.matmul(pr_onehots.astype("float32"), old_embeds, transpose_b = True)
	qtised		= tf.reshape(qtised, (FLAT, *(encoded_out.shape[1:])))
	decoder		= vqvae.get_layer("decoder")
	samples		= decoder.predict(qtised)
	show_new_brains(priors, samples)

def main():
	"""Evaluate the saved models and visualise some outputs"""
	tr, te, val	= get_ttv()
	test		= normalise(te)
	train		= normalise(tr)
	vnce		= var(train)
	vqvae		= load_model(MODEL_PATH, custom_objects = {"VectorQuantiser": VectorQuantiser})
	validate_vqvae(vqvae, test)
	validate_pcnn(vqvae, vnce, test)

if __name__ == "__main__":
	main()
