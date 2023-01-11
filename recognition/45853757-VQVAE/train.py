import tensorflow as tf
from modules import *
from dataset import *

batch_size = 64
epochs = 100

def train_vqvae():
    # Load our data
    training_data, validation_data, testing_data, data_variance = load_data()

    # Construct and train our model
    vqvae_model = VQVAEModel(variance=data_variance, latent_dim=16, n_embeddings=128)
    vqvae_model.compile(optimizer=keras.optimizers.Adam())
    print(vqvae_model.summary())

    return vqvae_model.fit(training_data, training_data, validation_data=(validation_data, validation_data), epochs=epochs, batch_size=batch_size)
