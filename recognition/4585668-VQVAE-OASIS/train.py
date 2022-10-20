from dataset	import get_ttv, normalise
from modules	import *

from numpy			import var
from tensorflow		import keras

LATENT_DIMENSION_SIZE	= 16
NUM_EMBEDDINGS			= 128
BETA					= 0.25
EPOCHS					= 100
# Based on GTX1660Ti Mobile
BATCH_SIZE				= 128
OPTIMISER				= keras.optimizers.Adam()
MODEL_PATH				= "vqvae.h5"

def train_vqvae(train_set, train_vnce):
	"""Train up a VQVAE from a training dataset and serialise it to a h5

	train_set	- the dataset to train the VQVAE on

	return		- the trained VQVAE and its wrapper trainer object
	"""
	print(train_vnce)

	trainer		= Trainer(train_vnce, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
	trainer.compile(optimizer = OPTIMISER)
	training	= trainer.fit(train_set, epochs = EPOCHS, batch_size = BATCH_SIZE)
	vqvae		= trainer.vqvae

	# In terms of testing and validating for the vqvae, we have no testing and validation for the vqvae

	return vqvae, trainer

def main():
	tr, te, va		= get_ttv()
	vnce			= var(tr)
	norm_train		= normalise(tr)
	vqvae, trainer	= train_vqvae(norm_train, vnce)
	vqvae.save(MODEL_PATH)

if __name__ == "__main__":
	main()
