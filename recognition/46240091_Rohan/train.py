from modules import *

VQVAE_EPOCHS = 25
VQVAE_BATCHSIZE = 64

def vqvae_training(training_data, data_variance, latent_dims = 16, num_embeddings = 128):
    vqvae = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae.compile(optimizer=keras.optimizers.Adam())
    history = vqvae.fit(training_data, epochs=VQVAE_EPOCHS, batch_size=VQVAE_BATCHSIZE)
    return vqvae, history


def VQVAE_training_plot(vqvae_history):
  plt.plot(vqvae_history.history['reconstruction_loss'])
  plt.title('VQVAE Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Reconstruction Loss'])
  plt.show()