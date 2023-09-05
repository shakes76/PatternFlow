from dataset import *

# Encoder
def encoder_net(latent_dim=32):
  input = tf.keras.layers.Input(input_shape, name="encoder_in")

  # Convolutional layers
  net = tf.keras.layers.Conv2D(depth, 3, padding='same', strides=2, activation='relu')(input)
  net = tf.keras.layers.Conv2D(depth*2, 3, padding='same', strides=2, activation='relu')(net)

  enc_out = tf.keras.layers.Conv2D(latent_dim, 1, padding='same')(net)
  
  return tf.keras.Model(inputs=input, outputs=enc_out, name='encoder')


# Build decoder
def decoder_net(latent_dim=32):
  decoder_in = tf.keras.Input(shape=encoder_net(latent_dim).output.shape[1:])

  net = tf.keras.layers.Conv2DTranspose(depth*2, 3, padding='same', strides=2, activation='relu')(decoder_in)
  net = tf.keras.layers.Conv2DTranspose(depth, 3, padding='same', strides=2, activation='relu')(net)
  network = tf.keras.layers.Conv2DTranspose(1, 3, padding='same')(net)

  return tf.keras.Model(inputs=decoder_in, outputs=network, name='decoder')


# Build Vector Quantizer for a discrete latent space
# Source: https://keras.io/examples/generative/vq_vae/
class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # Initialize embeddings for quantization 
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, input):
        # Calculate the input shape of the inputs and flatten inputs
        input_shape = tf.shape(input)
        flattened = tf.reshape(input, [-1, self.embedding_dim])

        # Vector Quantization for building discrete latent space via one-hot
        # encoding
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - input) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(input)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator (ignores derivative of theshhold function
        # and passes input gradient as an identity function)
        quantized = input + tf.stop_gradient(quantized - input)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between inputs and codes
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Calculate encoding indices for min distances
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

# VQVAE model
def vq_vae(latent_dim=32, embedding_num=128):
  vq_layer = VectorQuantizer(embedding_num, latent_dim, name='vector_quantise')
  encoder = encoder_net(latent_dim)
  decoder = decoder_net(latent_dim)
  inputs = tf.keras.Input(input_shape)
  enc = encoder(inputs)
  quantised_vae = vq_layer(enc)
  recon = decoder(quantised_vae)
  return tf.keras.Model(inputs, recon, name='vq_vae')