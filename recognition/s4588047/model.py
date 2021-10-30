import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

"""
Reference:
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    Tom Hennigan, VectorQuantizer layer 
    
    https://keras.io/examples/generative/vq_vae/
    Sayak Paul, Vector-Quantized Variational Autoencoders

"""

class VectorQuantizer(layers.Layer):
    '''
    Teh Vector Quantizer layer to encapsulate the vector quantizer logic along
    with embedding table that is then initialized to learn a codebook and
    return a quantized representatin of the image

    '''
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
        '''
        Gets the code indices in the codebook and returns them in an array
        
                Parameters:
                        flattened_inputs: a flatten shape of input 1D array
        
                Returns:
                        encoding_indices: a 1D array representing the codebook
                        representation
        '''
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
    '''
    The residual stack to keep information from the original array after the
    Conv2D operation and concatinate it with the after array
    4 * (Conv2D -> relu -> Conv2D -> relu)

    '''
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
            h_i = self._c2(h_i)
            h += h_i
        return tf.keras.activations.relu(h)        


class Encoder(layers.Layer):
    '''
    A Decoder for the network
    Conv2D -> Conv2D -> Conv2D -> ResidualStack

    '''
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
    '''
    A Decoder for the network
    Conv2D -> ResidualStack -> Conv2DTranpose -> Conv2DTranpose

    '''
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


class PixelConvLayer(layers.Layer):
    '''
    The first layer is the PixelCNN layer. This layer simply
    builds on the 2D convolutional layer, but includes masking.
    
    '''
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = tf.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)
    
    
class ResidualBlock(layers.Layer):
    '''
    This is just a normal residual block, but based on the PixelConvLayer.
    
    '''
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
