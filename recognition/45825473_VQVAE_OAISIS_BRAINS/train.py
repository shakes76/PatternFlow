import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import History
from keras import layers
from modules import VQVAETrainer, PixelConvLayer, ResidualBlock
from dataset import get_image_slices
from matplotlib import plot as plt
import tensorflow_probability as tfp

def train_vqvae(train_images):
  """
  Function that trains the VQVAE & Plots the respective losses as well as SSIM.
  Note the total loss was moved to another plot as initial Values for the total loss
  were significantly larger than the initial values for SSIM and reconstruction Loss.
  """
  data_variance = np.var(train_images)
  vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=32) #Reduced num_embeddings to resolve memory errors
  vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
  vqvae_trainer.fit(train_images, epochs=75, batch_size=32) #Training for batch=32 and 30 epochs
  
  #Plott losses post training
  loss = vqvae_trainer.history.history['loss']
  reconstruction_loss = vqvae_trainer.history.history['reconstruction_loss']
  epoch_ssim = vqvae_trainer.history.history['epoch_ssim']

  epochs = range(75)
  plt.plot(epochs, loss)
  plt.xlabel("Number of Epochs")
  plt.ylabel("Loss")
  plt.title("Total loss ")
  plt.show()
  plt.xlabel("Number of Epochs")
  plt.title("SSIM & Reconstruction Loss")
  plt.gca().legend(('Reconstruction Loss','SSIM'))
  plt.plot(epochs, reconstruction_loss, epochs, epoch_ssim )
  plt.show()

  return vqvae_trainer

def construct_and_train_pixelCNN(encoder, quantizer, vqvae_trainer, train_images):
  #This was increased as a part of 
  num_residual_blocks = 6 
  num_pixelcnn_layers = 6
  
  pixelcnn_input_shape = encoder.output.shape[1:-1]
  pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
  ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
  x = PixelConvLayer(
      mask_type="A", filters=256, kernel_size=9, activation="relu", padding="same")(ohe)

  for _ in range(num_residual_blocks):
      x = ResidualBlock(filters=256)(x)
  for _ in range(num_pixelcnn_layers):
      x = PixelConvLayer(mask_type="B", filters=128, kernel_size=1, strides=1,
          activation="relu", padding="valid")(x)
  out = keras.layers.Conv2D(  
      filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid")(x)
  pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")

  encoded_outputs = encoder.predict(train_images)
  flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
  codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
  codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

  # Compile and run the pixel_cnn model
  pixel_cnn.compile(optimizer=keras.optimizers.Adam(3e-4),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"])
  pixel_cnn.fit(x=codebook_indices, y=codebook_indices, batch_size=128, epochs=250, validation_split=0.05)

  inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
  outputs = pixel_cnn(inputs, training=False)
  categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
  outputs = categorical_layer(outputs)
  sampler = keras.Model(inputs, outputs)
  return pixel_cnn, sampler

def generate_probabilities_for_samples(pixel_cnn, sampler):
    # Creatation of an empty array of priors
  batch = 32
  priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
  batch, rows, cols = priors.shape
  print(f"Prior shape: {priors.shape}") #Check the shape before appending

  # Iterate over the 2D array
  for row in range(rows):
      for col in range(cols):
          probs = sampler.predict(priors)
          # Use the probabilities to pick pixel values and append the values to the priors.
          priors[:, row, col] = probs[:, row, col]

  return priors

def main():
  train_images, test_images, validate_images = get_image_slices()
  vqvae_trainer = train_vqvae(train_images)
  trained_vqvae_model = vqvae_trainer.vqvae

  encoder = vqvae_trainer.vqvae.get_layer("encoder")
  quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

  encoded_outputs = encoder.predict(test_images)
  flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
  codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
  codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

  pixel_cnn, sampler = construct_and_train_pixelCNN(encoder, quantizer, vqvae_trainer, train_images)
  priors = generate_probabilities_for_samples(pixel_cnn, sampler)

  #Save the trained decoder model after finishing all training.
  trained_vqvae_model.save("VQVAE_Model")
  return vqvae_trainer, quantizer, priors, encoded_outputs, pixel_cnn, sampler

vqvae_trainer, quantizer, priors, encoded_outputs = main()
