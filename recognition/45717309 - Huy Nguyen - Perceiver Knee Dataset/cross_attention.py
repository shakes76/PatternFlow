from tensorflow.keras import layers
import tensorflow as tf 

"""
Create a cross attention layer, with structure mirroring the paper's spec

Params:
    latent_size: int, size of the latent dimension
    data_size: int, rows * cols (total number of pixel in an image)
    proj_size: int, embedding size of fourier features, applied to
                    each element in the data and latent arrays
                    
Returns:
    a cross-attention model, taking in an img and a latent array 
    and outputing QKV cross-attention
"""
def cross_attention_layer(latent_size, data_size, proj_size):
    # projection_dim = data (1) + 2 * (2*bands + 1)
    # Input processed with a norm layer
    input_latent = layers.Input((latent_size, proj_size))
    latent_array = layers.LayerNormalization()(input_latent)

    input_data = layers.Input((data_size, proj_size))
    data_array = layers.LayerNormalization()(input_data)

    # QKV cross attention
    # K and V are projections of the input byte array, Q is a projection of a learned latent array
    q = layers.Dense(proj_size)(latent_array)
    k = layers.Dense(proj_size)(data_array)
    v = layers.Dense(proj_size)(data_array)
    
    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention = layers.Attention(use_scale=True)([q, k ,v])
    
    # pass to a linear layer
    attention = layers.Dense(proj_size)(attention)
    # Add input to output
    attention = layers.Add()([attention, latent_array])

    # Normalize
    attention = layers.LayerNormalization()(attention)

    # Pass the attention to a Dense block (MLPs)
    outputs = layers.Dense(proj_size, activation=tf.nn.gelu)(attention)

    # Final linear layer
    outputs = layers.Dense(proj_size)(outputs)

    # Add input to output
    outputs = layers.Add()([outputs, attention])

    # Create the Keras model and return it
    return tf.keras.Model(inputs=[input_latent, input_data], outputs=outputs)