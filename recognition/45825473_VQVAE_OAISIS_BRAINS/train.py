import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import History
import modules
import dataset

def train_vqvae(train_images):
  data_variance = np.var(train_images)
  vqvae_trainer = modules.VQVAETrainer(data_variance, latent_dim=16, num_embeddings=32) #Reduced num_embeddings to resolve memory errors
  vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
  vqvae_trainer.fit(train_images, epochs=10, batch_size=32) #Training for batch=32 and 30 epochs
  return vqvae_trainer

def main():
  train_images, test_images, validate_images = dataset.get_image_slices()
  vqvae_trainer = train_vqvae(train_images)
  trained_vqvae_model = vqvae_trainer.vqvae
  print(vqvae_trainer.history['val_loss'])  

  encoder = vqvae_trainer.vqvae.get_layer("encoder")
  quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

main()