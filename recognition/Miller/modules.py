"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""

import tensorflow as tf

"""Create Structure of VQ-VAR Model, set training paramters, train the model"""
# Vector Quantizer -> layer between encoder and decoder. Takes input from encoder and flattens. Then creates codebook
def vq_layer(embedding_num, latent_dimension):
    pass

"""

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Encoder Component
def encoder_component(latent_dimension):
    
    # Instantiate a lower level keras tensor to start building model of known input shape size (not including batch size)
    inputs = tf.keras.Input(shape=(256,256,1))

    #2D Convolutional Layers
        # filters -> dimesion of output space
        # kernal_size -> convolution window size
        # activation -> activation func used
            # relu ->
        # strides -> spaces convolution window moves vertically and horizontally 
        # padding -> "same" pads with zeros to maintain output size same as input size
    layer = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    layer = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(layer)
    
    outputs = tf.keras.layers.Conv2D(latent_dimension, 1, padding="same")(layer)
    
    return tf.keras.Model(inputs, outputs, name="encoder_component")
    

"""

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Decoder Component
def decoder_component(latent_dimension):

    # Instantiate a lower level keras tensor to start building model of known input shape size (not including batch size)
    inputs = tf.keras.Input(shape=encoder_component(latent_dimension).output.shape[1:])
    
    #Transposed Convolutional Layers (deconvolution)
        # filters -> dimesion of output space
        # kernal_size -> convolution window size
        # activation -> activation func used
            # relu ->
        # strides -> spaces convolution window moves vertically and horizontally 
        # padding -> "same" pads with zeros to maintain output size same as input size
    layer = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(inputs)
    layer = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(layer)
    
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, padding="same")(layer)
    
    return tf.keras.Model(inputs, outputs, name="decoder_model")


# Create a model instance and sets training paramters 
def vqvae_model(latent_dimension, embedings_num):

    # Instantiate a lower level keras tensor to start building model of known input shape size (not including batch size)
    inputs = tf.keras.Input(shape=(256, 256, 1))

    """Build Model Levels"""
    # Get encoder component layer with given latent dimension
    encoder = encoder_component(latent_dimension)
    encoder_component_outputs = encoder(inputs)

    # Get Quantized Layer with given number of embeddings and latent dimension
    quantized_layer = vq_layer(embedings_num, latent_dimension)
    quantized_latents_dimensions = quantized_layer(encoder_component_outputs)

    # Get decoder component layer with given latent dimension
    decoder = decoder_component(latent_dimension)
    reconstructions = decoder(quantized_latents_dimensions)

    return tf.keras.Model(inputs, reconstructions, name="vqvae_model")
