from base64 import encode
import tensorflow as tf


# https://keras.io/examples/generative/vq_vae/#vectorquantizer-layer
class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(initial_value = w_init(shape = (self.embedding_dim, self.num_embeddings), dtype = "float32"), trainable = True)

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b = True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis = 1, keepdims = True)
            + tf.reduce_sum(self.embeddings ** 2, axis = 0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis = 1)
        return encoding_indices


class VAE:
    def __init__(self, num_embeddings, latent_dim, beta):

        # Variables
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.beta = beta

    def encoder(self):
        inputs = tf.keras.Input(shape = (256, 256, 1))
        outputs = tf.keras.layers.Conv2D(32, 3, activation = "relu", strides = 2, padding = "same")(inputs)
        outputs = tf.keras.layers.Conv2D(64, 3, activation = "relu", strides = 2, padding = "same")(outputs)
        outputs = tf.keras.layers.Conv2D(self.latent_dim, 1, padding = "same")(outputs)
        model = tf.keras.Model(inputs, outputs, name = "encoder")
        return model, model.output.shape[1:]

    def vq_layer(self):
        return VectorQuantizer(self.num_embeddings, self.latent_dim, self.beta, name = "vector_quantizer")

    def decoder(self, encoder_output_shape):
        inputs = tf.keras.Input(shape = encoder_output_shape)
        outputs = tf.keras.layers.Conv2DTranspose(64, 3, activation = "relu", strides = 2, padding = "same")(inputs)
        outputs = tf.keras.layers.Conv2DTranspose(32, 3, activation = "relu", strides = 2, padding = "same")(outputs)
        outputs = tf.keras.layers.Conv2DTranspose(1, 3, padding = "same")(outputs)
        return tf.keras.Model(inputs, outputs, name = "decoder")
    
    def generate_model(self):
        encoder_model, encoder_output_shape = self.encoder()
        vq_layer_model = self.vq_layer()
        decoder_model = self.decoder(encoder_output_shape)

        inputs = tf.keras.Input(shape = (256, 256, 1))
        outputs = encoder_model(inputs)
        outputs = vq_layer_model(outputs)
        outputs = decoder_model(outputs)

        return tf.keras.Model(inputs, outputs, name = "VQ VAE model")