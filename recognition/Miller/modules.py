"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""
import tensorflow as tf

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""CREATE STRUCTURE OF VQ-VAR MODEL"""


"""
Class Representation of the Vector Quantization laye

Structure is: 
    1. Reshape into (n,h,w,d)
    2. Calculate L2-normalized distance between the inputs and the embeddings. -> (n*h*w, d)
    3. Argmin -> find minimum distance between indices for each n*w*h vector
    4. Index from dictionary: index the closest vector from the dictionary for each of n*h*w vectors
    5. Reshape into original shape (n, h, w, d)
    6. Copy gradients from q -> x
"""
class vq_layer(tf.keras.layers.Layer):
    def __init__(self, embedding_num, latent_dimension, beta, **kwargs):
        super().__init__(**kwargs)
        self.embedding_num = embedding_num
        self.latent_dimension = latent_dimension
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        initial = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(initial_value=initial((self.latent_dimension, self.embedding_num), dtype="float32"),trainable=True)

    def call(self, x):
        # Calculate the input shape and store for later -> Shape of (n,h,w,d)
        input = tf.shape(x)

        # Flatten the inputs to keep the embedding dimension intact. 
        # Combine all dimensions into last one 'd' -> (n*h*w, d) 
        flatten = tf.reshape(x, [-1, self.latent_dimension])

        # Get code indices
        # Calculate L2-normalized distance between the inputs and the embeddings.
        # For each n*h*w vectors, we calculate the distance from each of k vectors of embedding dictionaty to obtain matrix of shape (n*h*w, k)
        similarity = tf.matmul(flatten, self.embeddings)
        distances = (tf.reduce_sum(flatten ** 2, axis=1, keepdims=True) + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity)

        # For each n*h*w vectors, find the indices of closest k vector from dictionary; find minimum distance.
        encoded_indices = tf.argmin(distances, axis=1)
        
        # Turn the indices into a one hot encoded vectors; index the closest vector from the dictionary for each n*h*w vector
        encodings = tf.one_hot(encoded_indices, self.embedding_num)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to its original input shape -> (n,h,w,d)
        quantized = tf.reshape(quantized, input)
        """
        # Calculate vector quantization loss and add that to the layer
        commitment_loss = tf.reduan((quantized - tf.stop_gradient(x)) ** 2)
        codebook_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)

        #self.add_loss(self.beta * commitment_loss + codebook_loss)
        """
        # Straight-through estimator.
        # Unable to back propragate as gradient wont flow through argmin. Hence copy gradient from qunatised to x
        quantized = x + tf.stop_gradient(quantized - x)
        
        return quantized

"""
Returns layered model for encoder architecture built from convolutional layers. 

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Encoder Component
def encoder_component(image_size, latent_dimension):

    # Create model for layers
    encoder = tf.keras.models.Sequential(name = "encoder")
    
    #2D Convolutional Layers
        # filters -> dimesion of output space
        # kernal_size -> convolution window size
        # activation -> activation func used
            # relu ->
        # strides -> spaces convolution window moves vertically and horizontally 
        # padding -> "same" pads with zeros to maintain output size same as input size
    encoder.add(tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"))
    encoder.add(tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))
    encoder.add(tf.keras.layers.Conv2D(latent_dimension, 1, padding="same"))
    
    return encoder
    

"""
Returns layered model for decoder architecture built from  tranposed convolutional layers. 

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Decoder Component
def decoder_component():

    # Create model for layers
    decoder = tf.keras.models.Sequential(name="decoder")

    #Transposed Convolutional Layers (deconvolution)
        # filters -> dimesion of output space
        # kernal_size -> convolution window size
        # activation -> activation func used
            # relu ->
        # strides -> spaces convolution window moves vertically and horizontally 
        # padding -> "same" pads with zeros to maintain output size same as input size
    decoder.add(tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
    decoder.add(tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
    decoder.add(tf.keras.layers.Conv2DTranspose(1, 3, padding="same"))
    
    return decoder


# Create a model instance and sets training paramters 
class vqvae_model(tf.keras.models.Sequential):
    def __init__(self, image_size, latent_dimension, embeddings_num, beta, **kwargs):
        
        super(vqvae_model, self).__init__(**kwargs)
        self.image_size = image_size
        self.latent_dimension = latent_dimension
        self.embeddings_num = embeddings_num
        self.beta = beta
        
        # Create the model sequentially
        input_layer = tf.keras.layers.InputLayer(input_shape=(image_size,image_size,1))
        vector_quantiser_layer = vq_layer(embeddings_num, latent_dimension, beta)
        encoder = encoder_component(image_size, latent_dimension)
        decoder = decoder_component()
        
        # Add components of model
        self.add(input_layer)
        self.add(encoder)
        self.add(vector_quantiser_layer)
        self.add(decoder)

latent_dimensions = 16
embeddings_number = 64
image_size = 256
# beta = [0.25, 2]
beta = 0.25
model = vqvae_model(image_size, latent_dimensions, embeddings_number, beta)
model.summary()