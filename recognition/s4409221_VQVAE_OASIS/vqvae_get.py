#!/usr/bin/env python

"""
VQVAE Model.

This script puts components of the VQVAE model.

This script requires that Tensorflow, Keras and Numpy be installed within the 
environment you are running this script in.

This file can also be imported as a module and contains the following
class/es and functions:
    * VectorQuantizer - This is a class to implement a layer as a vector-quantizer.
    * call - A function to compute the loss of the layer.
    * get_code_indices - A function which calculates L2-normalized distance between the inputs and the embeddings.
    * get_encoder -  A function which implements the encoder of the VQ-VAE model.
    * get_decoder -   A function which implements the decoder of the VQ-VAE model.
    * get_vqvae -   A function which implements the VQ-VAE model.
    
    * VQVAETrainer -    A class which trains the VQ-VAE model.
    * metrics -   A function to define metrics for model performance.
    * train_step - A function to train the input data step by step and calculate the metrics accordingly.



"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

__author__      = "Gayatri Chaudhari"
__email__ = "s4409221@student.uq.edu.au"


#vectoriser

class VectorQuantizer(layers.Layer):    

    """
      This is a class to implement a layer as a vector-quantizer.

      Attributes: 
      Layer: This is a layer of the VQ-VAE model which is a Tensor object.
      
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        
        """
        The constructor for VectorQuantizer class.

        Parameters:
          num_embeddings (int): the number of embeddings for the vector quantizer.
          embedding_dim (int) : dimension of embedding.
          beta (float) : weights for each layer in vector quantizer.
        """

        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
          beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
          initial_value=w_init(
              shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
          ),
          trainable=True,
          name="embeddings_vqvae",
        ) 
  
    def call(self, x):
        """
        A function to compute the loss of the layer.

        Parameters:
          x (Tensor object): the inputs to the vector quantizer layer.
        Returns:
          quantized (Tensor object): output from vector quantizer layer 

        """
        #Calculate the input shape of the inputs and then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. 
        commitment_loss = self.beta * tf.reduce_mean(
          (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized
   

    def get_code_indices(self, flattened_inputs):
        """
        A function which calculates L2-normalized distance between the inputs and the embeddings.

        Parameters:
          flattened_inputs (Tensor object): takes in flattened input from the layers.

        Returns:
          encoding_indices (int) : returns the index containing the minimum distance.
          
        """
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
          tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
          + tf.reduce_sum(self.embeddings ** 2, axis=0)
          - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

#encoder 
def get_encoder(latent_dim=32):
    """
    A function which implements the encoder of the VQ-VAE model.

    Parameters:
    latent_dim (int): accepts the latent dimension.

    Returns:
    encoder (keras.Model) : the encoder of the VQ-VAE model

    """
    encoder_inputs = keras.Input(shape=(80, 80, 3))
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(
      encoder_inputs
    )
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 3, padding="same")(x)

    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

#decoder
def get_decoder(latent_dim=32):
    """
    A function which implements the decoder of the VQ-VAE model.

    Parameters:
    latent_dim (int): accepts the latent dimension.

    Returns:
    decoder (keras.Model) : the decoder of the VQ-VAE model

    """
    latent_inputs = keras.Input(shape=get_encoder().output.shape[1:])
    x = layers.Conv2DTranspose(16,3, activation="relu", strides=2, padding="same")(
      latent_inputs
    )
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same")(x)

    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

#Stand-alone VQ-VAE
def get_vqvae(latent_dim=32, num_embeddings=64):
    """
    A function which implements the VQ-VAE model.

    Parameters:
    latent_dim (int): accepts the latent dimension.
    num_embeddings (int): the number of embeddings in the vector quantizer.

    Returns:
    vq_vae (keras.Model) : the VQ-VAE model

    """
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(80, 80, 3))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)


    return keras.Model(inputs, reconstructions, name="vq_vae")

#VQ-VAE Trainer
class VQVAETrainer(keras.models.Model):
    """
    A class which trains the VQ-VAE model.

    Attributes:
    vq_vae: builds the VQ-VAE model for training data.

    """
    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        """
        The constructor for VQVAETrainer class.

        Parameters:
          train_variance (float): the variance of the input data.
          latern_dim (int) : number of latent dimensions.
          num_embeddings (int) : number of embeddings in vector quantizer of VQVAE model.
          
        """
      
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property

    def metrics(self):
        """
          A function to define metrics for model performance.

          Returns:
          total_loss_tracker (float) : returns the total loss of the current state of the model
          reconstruction_loss (float) : returns the reconstruction loss of the current state of the model
          vq_loss_tracker (float) : returns the vq_loss_tracker of the current state of the model
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        """
            A function to train the input data step by step and calculate the metrics accordingly.
            
            Parameters:
            x (array #change) : takes in the input training data 

            Returns:
            total_loss_tracker (float) : returns the total loss of the current state of the model
            reconstruction_loss (float) : returns the reconstruction loss of the current state of the model
            vq_loss_tracker (float) : returns the vq_loss_tracker of the current state of the model

        """

          
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            tmp_result = self.vqvae(x)

            #Calculate image difference using SSIM
            img_diff = 1-(tf.image.ssim(x,tmp_result,1.0))
                

            # Calculate the losses. Mean squared error, change
            reconstruction_loss = (
                tf.reduce_mean((x - tmp_result) ** 2) / self.train_variance
            )
            total_loss = ( img_diff + sum(self.vqvae.losses))

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        #self.optimizer = tf.keras.optimizers.Adam(0.001).minimize((-1*img_diff))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss(SSIM)": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }





