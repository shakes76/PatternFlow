from modules import *

VQVAE_EPOCHS = 25
VQVAE_BATCHSIZE = 64
PCNN_EPOCHS = 100
PCNN_BATCHSIZE = 128

def vqvae_training(training_data, data_variance, latent_dims = 16, num_embeddings = 128):
  vqvae = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
  vqvae.compile(optimizer=keras.optimizers.Adam())
  history = vqvae.fit(training_data, epochs=VQVAE_EPOCHS, batch_size=VQVAE_BATCHSIZE)
  return vqvae, history

def pcnn_training(pixel_cnn_model, codebook_indices):
  pixel_cnn_model.compile(optimizer=keras.optimizers.Adam(3e-4),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"],)
  history = pixel_cnn_model.fit(x=codebook_indices, y=codebook_indices, batch_size=128, epochs=100, validation_split=0.1)
  return pixel_cnn_model, history


def VQVAE_training_plot(vqvae_history):
  plt.plot(vqvae_history.history['reconstruction_loss'])
  plt.title('VQVAE Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Reconstruction Loss'])
  plt.show()


def PCNN_training_plot(pcnn_history):
  plt.plot(pcnn_history.history['loss'])
  plt.title('PIXELCNN Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training Loss'])
  plt.show()