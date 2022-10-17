import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import History
from keras import layers
from modules import VQVAETrainer, PixelConvLayer, ResidualBlock
from dataset import get_image_slices
import tensorflow_probability as tfp

def train_vqvae(train_images):
  data_variance = np.var(train_images)
  vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=32) #Reduced num_embeddings to resolve memory errors
  vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
  vqvae_trainer.fit(train_images, epochs=10, batch_size=32) #Training for batch=32 and 30 epochs
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
  pixel_cnn.fit(x=codebook_indices, y=codebook_indices, batch_size=128, epochs=100, validation_split=0.05)

  inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
  outputs = pixel_cnn(inputs, training=False)
  categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
  outputs = categorical_layer(outputs)
  sampler = keras.Model(inputs, outputs)
  return pixel_cnn, sampler

def generate_probabilities_for_samples(pixel_cnn, sampler):
    # Creatation of an empty array of priors
  batch = 10
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

  train_vqvae.save('VQVAE_Trainer')
  trained_vqvae_model.save("VQVAE_Model")
  ##TODO : Add saving the VQVAETrainer here. 
  ##TODO : ADD LOSS FUNCTION DETERMINATION/SAVING
  ##TODO : Add SSIM calculations
  return vqvae_trainer, quantizer, priors, encoded_outputs, pixel_cnn, sampler

vqvae_trainer, quantizer, priors, encoded_outputs = main()
