import numpy as np
import matplotlib.pyplot as plpt

from tensorflow import keras
from tensorflow.python.ops.unconnected_gradients import enum
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

class VectorQuantizer(layers.Layer):

    def __init__(self, num_embeddings, embedding_dim, beta = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The 'beta' parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize,
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value= w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name = "embeddings_vqvae",
        )

    def get_code_indices(self, flattened_inputs):
        #Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis = 1, keepdims = True)
            + tf.reduce_sum(self.embeddings ** 2, axis = 0)
            - 2 * similarity
        )

        # Derive the indices for minimum distance
        encoding_indices = tf.argmin(distances, axis = 1)
        return encoding_indices

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

def get_encoder(latent_dim=16, input_image_shape=(256,256,3)):

    #TODO change this input size to match that of data image dimension
    
    # OLD 
    #encoder_inputs = keras.Input(shape=(28, 28, 1))

    encoder_inputs = keras.Input(shape = (input_image_shape))

    #normalization_layer = tf.keras.layers.Rescaling(1./255)
    #x = encoder_inputs(normalization_layer)

    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

def get_decoder(latent_dim=16):
    #print("get_encoder(latent_dim).output.shape[1:] = ", get_encoder(latent_dim).output.shape[1:])
    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
    
    #print("decoder_outputs.shape", decoder_outputs.shape)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VQVAE(keras.models.Model):

    def __init__(self, latent_dim = 16, num_embeddings = 64, image_shape = (256,256,3) ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.image_shape = image_shape
        self.normalization_layer = tf.keras.layers.Rescaling(1./255.)
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.latent_dim, name='vector_quantizer')
        self.encoder = get_encoder(self.latent_dim,  self.image_shape)
        self.decoder = get_decoder(self.latent_dim)
    
    def call(self, x):
        x = self.normalization_layer(x)
        x = self.encoder(x)
        x = self.vq_layer(x)
        return self.decoder(x)


    def save_model(self, path):
        super.save_model(path)

class VQVAETrainer(keras.models.Model):

    def __init__(self, train_variance, latent_dim=32, num_embeddings=128, image_shape = (256,256,3), **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.image_shape = image_shape

        #self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, self.image_shape)
        self.vqvae = VQVAE(self.latent_dim, self.num_embeddings, self.image_shape)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]
    
    def call(self, x):
        return

    @tf.function
    def train_step(self, x):
    
        with tf.GradientTape() as tape:

            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)
            #print(reconstructions)
            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # A Conv2D layer to initialize kernel variables
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

class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

def get_pixel_cnn(vqvae_trainer, pixelcnn_input_shape, num_pixelcnn_layers, num_residual_blocks):
    
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
    )(ohe)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = keras.layers.Conv2D(
        filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    return pixel_cnn

# def get_vqvae(latent_dim=16, num_embeddings=64):
#     vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
#     encoder = get_encoder(latent_dim)
#     decoder = get_decoder(latent_dim)
#     inputs = keras.Input(shape=(28, 28, 1))
#     encoder_outputs = encoder(inputs)
#     quantized_latents = vq_layer(encoder_outputs)
#     reconstructions = decoder(quantized_latents)
#     return keras.Model(inputs, reconstructions, name="vq_vae")
