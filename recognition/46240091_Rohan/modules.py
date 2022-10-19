from glob import glob
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import os
from typing import Tuple
import keras.models
import keras.preprocessing.image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import tensorflow_probability as tfp
from keras import layers


def encoder(latent_dim=16):
  """
  Encoder Network for VQVAE
  """
  model = keras.Sequential(name="encoder");
  model.add(keras.Input(shape=(128, 128, 1)))
  model.add(layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2D(latent_dim, 1, padding="same"))
  return model


def decoder(latent_dim=16):
  """
  Decoder Network for VQVAE
  """
  model = keras.Sequential(name="decoder");
  model.add(keras.Input(shape=encoder().output.shape[1:]))
  model.add(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
  model.add(layers.Conv2DTranspose(1, 3, padding="same"))
  return model

"""
VQVAE model based on the paper https://arxiv.org/abs/1711.00937
VectorQuantizer class from Keras tutorial on VQVAE https://keras.io/examples/generative/vq_vae/#vectorquantizer-layer
"""

class VectorQuantizer(layers.Layer):
  """
  Vector Quantizer custom layer according to https://keras.io/examples/generative/vq_vae/#vectorquantizer-layer
  """

  def __init__(self, num_embeddings, embedding_dim, **kwargs):
      super().__init__(**kwargs)
      self.embedding_dim = embedding_dim
      self.num_embeddings = num_embeddings
      #Beta = 1 found to give very good results in less number of epochs, should be kept between 0.25 and 2
      self.beta = 1
      
      #Initialising the embeddings randomly, which will be quantized and then L2 distance calculated between output from encoder
      #and entries from this.
      initial = tf.random_uniform_initializer()
      self.embeddings = tf.Variable(initial_value=initial(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
          trainable=True,
          name="embeddings_vqvae",
      )

  def get_code_indices(self, flattened_inputs):
    """
    Calculating the L2 normalized distance between flattened input and codebook entries
    and then returning the indices with min distance
    """
    similarity = tf.matmul(flattened_inputs, self.embeddings)
    distances = (
        tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
        + tf.reduce_sum(self.embeddings ** 2, axis=0)
        - 2 * similarity
    )

    encoding_indices = tf.argmin(distances, axis=1)
    return encoding_indices

  def call(self, x):
      #Input shape calculated and inputs reshaped (embedding dimesions not flattened as it is size of latent embeddings)
      input_shape = tf.shape(x)
      flattened = tf.reshape(x, [-1, self.embedding_dim])
      
      #Applying one-hot encoding to code with min distance from flattened inputs
      encoding_indices = self.get_code_indices(flattened)
      encodings = tf.one_hot(encoding_indices, self.num_embeddings)
      quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
      quantized = tf.reshape(quantized, input_shape)

      commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2) 
      codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
      self.add_loss(commitment_loss + codebook_loss)
      
      #Straight through estimator between decoder and encoder.
      quantized = x + tf.stop_gradient(quantized - x)
      return quantized


def vqvae_model(latent_dim=16, num_embeddings=64):
  """
  Vqvae model that combines the encoder, vq-layer, and decoder 
  """
  encoder_model = encoder(latent_dim)
  decoder_model = decoder(latent_dim)
  encoder_inputs = keras.Input(shape=(128, 128, 1))
  vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
  vq_layer_inputs = encoder_model(encoder_inputs)
  decoder_inputs = vq_layer(vq_layer_inputs)
  vqvae_output = decoder_model(decoder_inputs)
  return keras.Model(encoder_inputs, vqvae_output, name="vq_vae")      