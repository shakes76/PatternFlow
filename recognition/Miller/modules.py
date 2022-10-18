"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""
import tensorflow as tf
from tensorflow.python.keras.engine import input_spec

"""CREATE STRUCTURE OF VQ-VAR MODEL"""

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
        # Calculate the input shape
        input = tf.shape(x)
        print(input)
        print("ahhh")
        

        # Flatten the inputs to keep the embedding dimension intact.
        flatten = tf.reshape(x, [-1, self.latent_dimension])

        # Get code indices
        # Calculate L2-normalized distance between the inputs and the embeddings.
        similarity = tf.matmul(flatten, self.embeddings)
        distances = (tf.reduce_sum(flatten ** 2, axis=1, keepdims=True) + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity)

        # Derive the indices for minimum distances.
        encoded_indices = tf.argmin(distances, axis=1)
        
        # Turn the indices into a one hot encoded vectors
        encodings = tf.one_hot(encoded_indices, self.embedding_num)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to its original input shape
        quantized = tf.reshape(quantized, input)
        """
        # Calculate vector quantization loss and add that to the layer
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        
        #self.add_loss(self.beta * commitment_loss + codebook_loss)
        """
        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)

        return quantized


"""
def initialise_embeddings(embedding_num, latent_dimension):
    initial = tf.random_uniform_initializer()
    return tf.Variable(initial_value=initial(shape=(embedding_num, latent_dimension), dtype="float32"), trainable=True, name="embeddings")

# Vector Quantizer -> layer between encoder and decoder. Takes input from encoder and flattens. Then creates codebook
def vq_layer(embedding_num, latent_dimension, beta, x):
    
    # Initialize the embeddings which will be quantized.
    embeddings = initialise_embeddings(embedding_num, latent_dimension)
    
    # Calculate the input shape
    input = tf.keras.layers.shape(x)
    print(input)
    print("ahhhh")

    # Flatten the inputs to keep the embedding dimension intact.
    flatten = tf.reshape(x, [-1, latent_dimension])

    # Get code indices
    # Calculate L2-normalized distance between the inputs and the embeddings.
    similarity = tf.matmul(flatten, embeddings)
    distances = (tf.reduce_sum(flatten ** 2, axis=1, keepdims=True) + tf.reduce_sum(embeddings ** 2, axis=0) - 2 * similarity)

    # Derive the indices for minimum distances.
    encoded_indices = tf.argmin(distances, axis=1)
       
    # Turn the indices into a one hot encoded vectors
    encodings = tf.one_hot(encoded_indices, embedding_num)
    quantized = tf.matmul(encodings, embeddings, transpose_b=True)

    # Reshape the quantized values back to its original input shape
    quantized = tf.reshape(quantized, input)

    # Calculate vector quantization loss and add that to the layer
    commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
    codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
    
    #self.add_loss(self.beta * commitment_loss + codebook_loss)

    # Straight-through estimator.
    quantized = x + tf.stop_gradient(quantized - x)

    return quantized
"""


"""

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Encoder Component
def encoder_component(image_size, latent_dimension):

    encoder = tf.keras.models.Sequential(name = "encoder")
    # Instantiate a lower level keras tensor to start building model of known input shape size (not including batch size)
    """inputs = tf.keras.layers.Input(shape=(image_size, image_size, 1 ), batch_size=1)
    print(inputs.shape)"""
    
    #2D Convolutional Layers
        # filters -> dimesion of output space
        # kernal_size -> convolution window size
        # activation -> activation func used
            # relu ->
        # strides -> spaces convolution window moves vertically and horizontally 
        # padding -> "same" pads with zeros to maintain output size same as input size
    encoder.add(tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", input_shape=(image_size,image_size,1)))
    encoder.add(tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))

    encoder.add(tf.keras.layers.Conv2D(latent_dimension, 1, padding="same"))
    
    #return tf.keras.Model(inputs, outputs, name="encoder_component")
    return encoder
    

"""

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Decoder Component
def decoder_component(image_size):

    # Instantiate a lower level keras tensor to start building model of known input shape size (not including batch size)
    """inputs = tf.keras.Input((encoder_shape), batch_size=1)
    print(inputs.shape)"""
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
    
    #return tf.keras.Model(inputs, outputs, name="decoder_model")
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
        vector_quantiser_layer = vq_layer(embeddings_num, latent_dimension, beta)
        encoder = encoder_component(image_size, latent_dimension)
        decoder = decoder_component(image_size)

        # Add components of model
        self.add(encoder)
        self.add(vector_quantiser_layer)
        self.add(decoder)

    
    """print("START")
    # Instantiate a lower level keras tensor to start building model of known input shape size (not including batch size)
    inputs = tf.keras.layers.Input((image_size, image_size, 1 ), batch_size=1)


    #encoder_shape = inputs.shape[1:]
    print("The original inputs: ", inputs.shape)

    """
    #Build Model Levels
    """
    # Get encoder component layer with given latent dimension
    encoder = encoder_component(image_size, latent_dimension)

    #print("The encoder shape is: ", encoder.shape)
    #exit()
    #encoder_component_outputs = encoder(inputs)
    #print(encoder_component_outputs.shape)
    
    # Get Quantized Layer with given number of embeddings and latent dimension
    quantized_layer = vq_layer(embedings_num, latent_dimension, beta)
    quantized_latents_dimensions = quantized_layer(encoder)

    # Get decoder component layer with given latent dimension
    decoder = decoder_component(image_size, latent_dimension, inputs.shape[1:])
    print(decoder.shape)
    reconstructions = decoder*(quantized_latents_dimensions)

    return tf.keras.Model(inputs, reconstructions, name="vqvae_model")
    """



