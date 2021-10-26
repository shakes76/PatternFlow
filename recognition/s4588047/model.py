import tensorflow as tf
from tensorflow.keras import layers

"""
Reference:
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    Tom Hennigan, VectorQuantizer layer 

"""

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
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
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x)**2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(x))**2)
        loss = q_latent_loss + self.beta * e_latent_loss
        
        self.add_loss(loss)

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


class ResidualStack(layers.Layer):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        
        self._c1 = tf.keras.layers.Conv2D(
            self._num_residual_hiddens,(3, 3),strides=(1, 1),padding="same",name="res3x3")
        self._c2 = tf.keras.layers.Conv2D(
            self._num_hiddens,kernel_size=(1, 1),strides=(1, 1),padding="same",name="res1x1")
        
    def call(self, x):
        h = x
        for i in range(self._num_residual_layers):
            h_i = tf.keras.activations.relu(h)
            h_i = self._c1(h_i)
            h_i = tf.keras.activations.relu(h_i)
            #print("ok h_i: {}".format(h_i.shape))

            h_i = self._c2(h_i)
            #print("ok h_i: {}".format(h_i.shape))
            h += h_i
        return tf.keras.activations.relu(h)        


class Encoder(layers.Layer):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = layers.Conv2D(
            self._num_hiddens // 2,
            (4, 4),
            strides=(2, 2),
            activation='relu',
            padding="same",
            name="enc_1")
        self._enc_2 = layers.Conv2D(
            self._num_hiddens,
            (4, 4),
            strides=(2, 2),
            activation='relu',
            padding="same",
            name="enc_2")
        self._enc_3 = layers.Conv2D(
            self._num_hiddens,
            (3, 3),
            strides=(1, 1),
            activation='relu',
            padding="same",
            name="enc_3")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def __call__(self, x):
        
        h = self._enc_1(x)
        print(h.shape)
        h = self._enc_2(h)
        h = self._enc_3(h)
        return self._residual_stack(h)
    
    
class Decoder(layers.Layer):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name=None):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = layers.Conv2D(
            self._num_hiddens,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="dec_1")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = layers.Conv2DTranspose(
            self._num_hiddens // 2,
            (4, 4),
            strides=(2, 2),
            activation='relu',
            padding="same",
            name="dec_2")
        self._dec_3 = layers.Conv2DTranspose(
            1,
            (4, 4),
            strides=(2, 2),
            padding="same",
            name="dec_3")

    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = self._dec_2(h)
        x_recon = self._dec_3(h)
        return x_recon
