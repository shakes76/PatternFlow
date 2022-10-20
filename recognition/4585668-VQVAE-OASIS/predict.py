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

RECONS_TO_VIEW	= 5
NOVELS			= 5
COLS			= 2
MAX_VAL			= 1.0
CENTRE			= 0.5
GREY			= cm.gray
BATCH			= 10

def compare(images, recons):
	ssims = 0
	for i, pair in enumerate(zip(images, recons)):
		o, r	= pair
		orig	= convert_image_dtype(o, tf.float32)
		recon	= convert_image_dtype(r, tf.float32)
		sim		= ssim(orig, recon, max_val = MAX_VAL)
		ssims	+= sim
		subplot(RECONS_TO_VIEW, COLS, COLS * i + 1)
		imshow(o + CENTRE, cmap = GREY)
		title("Test Input")
		axis("off")
		subplot(RECONS_TO_VIEW, COLS, COLS * (i + 1))
		imshow(r + CENTRE, cmap = GREY)
		title("Test Reconstruction")
		axis("off")
		suptitle("SSIM: %.2f" %sim)
	show()

	return ssims

def validate_vqvae(test):
	vqvae		= load_model(MODEL_PATH, custom_objects = {"VectorQuantiser": VectorQuantiser})
	image_inds	= choice(len(test), RECONS_TO_VIEW)
	images		= test[image_inds]
	recons		= vqvae.predict(images)
	ssim		= compare(images, recons)
	avg_ssim	= ssim / RECONS_TO_VIEW
	print(f"Average SSIM: {avg_ssim}")

def show_new_brains(priors, samples):
	for i in range(NOVELS):
		subplot(NOVELS, COLS, COLS * i + 1)
		imshow(priors[i], cmap = GREY)
		title("PCNN Prior")
		axis("off")
		subplot(NOVELS, COLS, COLS * (i + 1))
		imshow(samples[i] + CENTRE, cmap = GREY)
		title("Decoded Prior")
		axis("off")
	show()

def validate_pcnn(train_vnce, test):
	pcnn	= load_model(PCNN_PATH, custom_objects = {"PixelConvolution": PixelConvolution, "ResidualBlock": ResidualBlock})
	#trainer	= Trainer(train_vnce, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
	#trainer.build(inputs = ENC_IN_SHAPE)
	#trainer.build((None, *(ENC_IN_SHAPE)[:-1]))
	#trainer.compile(optimizer = OPTIMISER())
	#trainer.build(inputs = ENC_IN_SHAPE, outputs = ENC_IN_SHAPE)
	#trainer.load_weights(TRAINER_PATH)
	priors	= zeros(shape = (BATCH,) + (pcnn.input_shape)[1:])
	batch, rows, columns = priors.shape

	for r in range(rows):
		for c in range(columns):
			logs	= pcnn.predict(priors)
			sampler	= tfp.distributions.Categorical(logs)
			prob	= sampler.sample()
			priors[:, r, c] = prob[:, r, c]

	vqvae		= load_model(MODEL_PATH, custom_objects = {"VectorQuantiser": VectorQuantiser})
	encoder		= vqvae.get_layer("encoder")
	encoded_out	= encoder.predict(test)
	old_embeds	= vqvae.get_layer("quantiser").embeddings
	pr_onehots	= tf.one_hot(priors.astype("int32"), NUM_EMBEDDINGS).numpy()
	qtised		= tf.matmul(pr_onehots.astype("float32"), old_embeds, transpose_b = True)
	qtised		= tf.reshape(qtised, (FLAT, *(encoded_out.shape[1:])))
	decoder		= vqvae.get_layer("decoder")
	samples		= decoder.predict(qtised)
	show_new_brains(priors, samples)

def main():
	tr, te, val	= get_ttv()
	test		= normalise(te)
	train		= normalise(tr)
	vnce		= var(train)
	#validate_vqvae(test)
	validate_pcnn(vnce, test)

if __name__ == "__main__":
	main()
