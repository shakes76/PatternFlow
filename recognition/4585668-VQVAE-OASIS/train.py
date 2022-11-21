"""
Build and train a VQVAE and PCNN
"""

from dataset			import get_ttv, normalise
from modules			import *

from matplotlib.pyplot	import plot, title, xlabel, ylabel, show, legend
from numpy				import var
from tensorflow			import keras

LATENT_DIMENSION_SIZE	= 8
NUM_EMBEDDINGS			= 64
BETA					= 0.25
VQVAE_EPOCHS			= 100
PCNN_EPOCHS				= 100
# Based on GTX1660Ti Mobile
BATCH_SIZE				= 128
OPTIMISER				= keras.optimizers.Adam
PCNN_OPT				= 3e-4
LOSS					= keras.losses.SparseCategoricalCrossentropy(from_logits = True)
VALIDATION_SPLIT		= 0.1
METRICS					= ["accuracy"]
MODEL_PATH				= "vqvae.h5"
PCNN_PATH				= "pcnn.h5"
LOC						= "upper right"

def train_vqvae(train_set, train_vnce):
	"""Train up a VQVAE from a training dataset and serialise it to a h5

	train_set	- the dataset to train the VQVAE on

	return		- the trained VQVAE and its wrapper trainer object
	"""

	trainer		= Trainer(train_vnce, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA)
	trainer.compile(optimizer = OPTIMISER())
	training	= trainer.fit(train_set, epochs = VQVAE_EPOCHS, batch_size = BATCH_SIZE)
	vqvae		= trainer.vqvae

	plot(training.history["reconstruction_loss"])
	title("VQVAE Training Loss per Epoch")
	ylabel("Loss")
	xlabel("Epoch")
	show()

	# In terms of testing and validating for the vqvae, we have no testing and validation for the vqvae

	return vqvae, trainer

def pcnn_graph(first, second, metric):
	"""Display a generic graph about the PCNN

	first	- first of two metrics to be graphed on the same plot
	second	- second of two metrics to be graphed on the same plot
	metric	- the Y axis of the plot
	"""

	plot(first)
	plot(second)
	title(f"PCNN {metric} per Epoch")
	ylabel(metric)
	xlabel("Epoch")
	legend(["Training", "Validation"], loc = LOC)
	show()

def pcnn_metrics(metrics):
	"""Display the losses and accuracy of the PCNN model

	metrics	- training metrics obtained from the PCNN
	"""

	tloss,	vloss	= metrics["loss"],		metrics["val_loss"]
	tacc,	vacc	= metrics["accuracy"],	metrics["val_accuracy"]
	pcnn_graph(tloss, vloss, "Loss")
	pcnn_graph(tacc, vacc, "Accuracy")

def main():
	"""Train and save a VQVAE and a PCNN, and view their training metrics"""
	tr, te, va		= get_ttv()
	vnce			= var(tr)
	train			= normalise(tr)
	test			= normalise(te)
	vqvae, trainer	= train_vqvae(train, vnce)
	vqvae.save(MODEL_PATH)
	encoder			= trainer.vqvae.get_layer("encoder")
	encoded_out		= encoder.predict(test)
	encoded_out		= encoded_out[:len(encoded_out) // 2] # Dump half of them because I have no memory left
	qtiser			= trainer.vqvae.get_layer("quantiser")
	flat_encs		= encoded_out.reshape(FLAT, encoded_out.shape[FLAT])
	codebooks		= qtiser.code_indices(flat_encs)
	codebooks		= codebooks.numpy().reshape(encoded_out.shape[:-1])
	pcnn			= build_pcnn(trainer, encoded_out)
	pcnn.compile(optimizer = OPTIMISER(PCNN_OPT), loss = LOSS, metrics = METRICS)
	pcnn_training	= pcnn.fit(x = codebooks, y = codebooks, batch_size = BATCH_SIZE, epochs = PCNN_EPOCHS, validation_split = VALIDATION_SPLIT)
	pcnn_metrics(pcnn_training.history)
	pcnn.save(PCNN_PATH)

if __name__ == "__main__":
	main()
