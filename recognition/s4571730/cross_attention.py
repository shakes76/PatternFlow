from tensorflow.keras import layers
import tensorflow as tf 
from dense_net import dense_block

def cross_attention_layer(latent_size, data_size, proj_size, dense_units):
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
    # outputs = dense_block(dense_units)(attention)
    outputs = layers.Dense(dense_units[0], activation=tf.nn.gelu)(attention)

    # Final linear layer
    outputs = layers.Dense(dense_units[-1])(outputs)

    # Add input to output
    outputs = layers.Add()([outputs, attention])

    # Create the Keras model.
    model = tf.keras.Model(inputs=[input_latent, input_data], outputs=outputs)
    return model