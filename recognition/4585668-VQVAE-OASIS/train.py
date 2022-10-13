#from dataset	import *
from modules	import *

from numpy			import var
#from os				import environ
from tensorflow		import keras
from keras.models	import load_model

LATENT_DIMENSION_SIZE	= 16
NUM_EMBEDDINGS			= 128
BETA					= 0.25
EPOCHS					= 30
# Based on GTX1660Ti Mobile
BATCH_SIZE				= 128
OPTIMIZER				= keras.optimizers.Adam()
MODEL_PATH				= "vqvae.h5"

def train_vqvae(train_set):
	"""Train up a VQVAE from a training dataset and serialise it to a h5

	train_set	- the dataset to train the VQVAE on

	return		- the trained VQVAE and its wrapper trainer object
	"""

	train_vnce	= var(train_set)
	trainer		= Trainer(train_vnce, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
	trainer.compile(optimizer = OPTIMIZER)
	training	= trainer.fit(train_set, epochs = EPOCHS, batch_size = BATCH_SIZE)
	vqvae		= trainer.vqvae
	#vqvae.save(MODEL_PATH)

	# In terms of testing and validating for the vqvae, we have no testing and validation for the vqvae

	return vqvae, trainer

def deserialise_vqvae():
	return load_model(MODEL_PATH)

# TESTING TODO DELETE
def main():
	from dataset import get_ttv
	tr, te, va = get_ttv()
	print(type(tr))
	vqvae, trainer = train_vqvae(tr)
	print()
	print(vqvae)
	print()
	print(trainer)

if __name__ == "__main__":
	main()
