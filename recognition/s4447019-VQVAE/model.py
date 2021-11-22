import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from skimage.metrics import structural_similarity

###############################################################
# Vector Quantizer class from Kera's tutorial
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (beta)
        # The commitment loss coefficient, beta, is best kept between [0.25, 2]
        # as per Neural Discrete Representation Learning (page 4)

        # Initialisation of embeddings to be quantised
        w_init = tf.random_uniform_initializer() #Initialisation of random values
        self.embeddings = tf.Variable( #Definition of the embeddings tensor variable
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
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
###############################################################
    
###############################################################
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
###############################################################

###############################################################
# VQVAE Trainer class from Kera's tutorial
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=16, num_embeddings=128, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance #Define the variance of the training image set
        self.latent_dim = latent_dim #Define the latent dimension of the model
        self.num_embeddings = num_embeddings #Define the total number of embeddings of the model

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings) #Call the model with the given hyperparameters
        
        # Define the reconstuction and total loss variables to be printed during training
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
            self.reconstruction_loss_tracker]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # These are the outputs from the VQ-VAE
            reconstructions = self.vqvae(x)

            # Calculate the total and reconstruction losses
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Application of backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Update the loss variables
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        # Print the loss results during the training
        return {"loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()}

###############################################################
    
###############################################################
# PIXELCNN from Kera's tutorial
# Reference: https://keras.io/examples/generative/vq_vae/
# Author: Sayak Paul

# PixelCNN layer class defined
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

# Residual block layer class defined
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")
        self.pixel_conv = PixelConvLayer(mask_type="B", filters=filters // 2, kernel_size=3, activation="relu", padding="same")
        self.conv2 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
###############################################################