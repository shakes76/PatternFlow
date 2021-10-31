import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

# Encoder and decoder network functions #
# Encoder network (inference/recognition model)
def get_encoder(latent_dim=16): #Define the latent dimension
    
    input_layer = keras.Input(shape=(128, 128, 1)) #Input layer
    conv2D = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_layer) #Convolutional layer 1
    conv2D = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(conv2D) #Convolutional layer 2
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(conv2D) #Output layer
    
    return keras.Model(input_layer, encoder_outputs, name="encoder")

# Decoder network (generative model)
def get_decoder(latent_dim=16):
    
    latent_inputs = keras.Input(shape=get_encoder().output.shape[1:]) #Input layer
    convTran = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs) #Transpose Convolutional layer 1
    convTran = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(convTran) #Transpose Convolutional layer 2
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(convTran) #Output layer
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Stand alone VQVAE #
def get_vqvae(latent_dim=16, num_embeddings=32):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(128, 128, 1)) #Define the input shape
    encoder_outputs = encoder(inputs) #Set encoder outputs to encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs) 
    # Need to ensure the output channels of the encoder match the latent dimension
    # of the VQ
    reconstructions = decoder(quantized_latents)
    
    return keras.Model(inputs, reconstructions, name="vq_vae")

# Print summary of stand alone VQ-VAE model
get_vqvae().summary()