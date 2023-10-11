from base64 import decode
import code
from matplotlib.cbook import flatten
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, ReLU, Add, Conv2D, Conv2DTranspose
import tensorflow_probability as tfp

class VectorQuantizer(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, name="VQ", **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # Initialise flattened embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), 
                dtype='float32'
            ),
            trainable=True,
            name=name
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, (-1, self.embedding_dim))

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        
        commitment_loss = tf.nn.l2_loss(tf.stop_gradient(quantized) - x)**2
        codebook_loss = tf.nn.l2_loss(quantized - tf.stop_gradient(x))**2

        self.add_loss(self.beta * commitment_loss + codebook_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        return quantized
        
    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


def resblock(x, filters=256):
    xconv = Conv2D(filters, 3, strides=1, activation='relu', padding='same')(x)
    xconv = Conv2D(x.shape[-1], 1, strides=1, padding='same')(xconv)
    out = Add()([x, xconv])
    return ReLU()(out)  

class PixelCNN(Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelCNN, self).__init__()
        self.mask_type = mask_type
        self.conv = Conv2D(**kwargs)
    
    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[:kernel_shape[0]//2, ...] = 1.0
        self.mask[kernel_shape[0]//2, :kernel_shape[1]//2, ...] = 1.0
        if self.mask == 'B':
            self.mask[kernel_shape[0]//2, kernel_shape[1]//2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

class ResidualBlock(Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters=filters, kernel_size=1, activation='leaky_relu')
        self.pixelcnn = PixelCNN(mask_type='B', filters=filters//2, kernel_size=3, activation='leaky_relu', padding='same')
        self.conv2 = Conv2D(filters=filters, kernel_size=1, activation='leaky_relu')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixelcnn(x)
        x = self.conv2(x)
        return tf.keras.layers.add([inputs, x])

def get_pixelcnn(input_shape, num_embeddings, filters=128, num_residual_blocks=2, num_pixelcnn_layers=2, **kwargs):
    pixelcnn_inputs = Input(shape=input_shape, dtype=tf.int32)
    onehot = tf.one_hot(pixelcnn_inputs, num_embeddings)
    x = PixelCNN(mask_type='A', filters=filters, kernel_size=32, activation='leaky_relu', padding='same')(onehot)
    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=filters)(x)
    for _ in range(num_pixelcnn_layers):
        x = PixelCNN(mask_type='B', filters=filters, kernel_size=1, strides=1, activation='leaky_relu', padding='valid')(x)
    out = Conv2D(filters=num_embeddings, kernel_size=1, strides=1, padding="valid")(x)
    return tf.keras.Model(pixelcnn_inputs, out, name='pixelcnn')


def get_vqvae(latent_dim=16, num_embeddings=64, input_shape=(256, 256, 1), residual_hiddens=64):
    latent_dim = latent_dim
    num_embeddings = num_embeddings
    
    # Build encoder
    encoder_in = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, activation='leaky_relu', padding='same')(encoder_in)
    x = Conv2D(64, 3, strides=2, activation='leaky_relu', padding='same')(x)
    encoder_out = Conv2D(latent_dim, 1, padding="same")(x)
    encoder = tf.keras.Model(encoder_in, encoder_out, name='encoder')

    # Build decoder
    decoder_in = Input(shape=encoder.output.shape[1:])
    y = Conv2DTranspose(64, 3, strides=2, activation='leaky_relu', padding='same')(decoder_in)
    y = Conv2DTranspose(32, 3, strides=2, activation='leaky_relu', padding='same')(y)
    decoder_out = Conv2DTranspose(1, 3, strides=1, activation='leaky_relu', padding='same')(y)
    decoder = tf.keras.Model(decoder_in, decoder_out, name='decoder')

    # Add VQ layer
    vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim, name='vq')

    inputs = Input(shape=input_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return tf.keras.Model(inputs, reconstructions, name='vq-vae')
        
def get_pixelcnn_sampler(pixelcnn):
    inputs = Input(shape=pixelcnn.input_shape[1:])
    outputs = pixelcnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    return tf.keras.Model(inputs, outputs)