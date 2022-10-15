"""
“modules.py" containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""

import tensorflow as tf

"""Create Structure of VQ-VAR Model, set training paramters, train the model"""
# Vector Quantizer -> layer between encoder and decoder. Takes input from encoder and flattens. Then creates codebook
#def vq_layer():

"""

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Encoder Component
#def encoder_component(latent_dimension):
    

"""

activations: ReLU advised as other activations are not optimal for encoder/decoder quantization architecture.
e.g. Leaky ReLU activated models are difficult to train -> cause sporadic loss spikes that model struggles to recover from
"""
# Decoder Component
#def decoder_component(latent_dimension):

# Create a model instance and sets training paramters 
#def vqvae_model():
