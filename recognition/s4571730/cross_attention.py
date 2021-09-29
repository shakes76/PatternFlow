from tensorflow.keras import layers
import tensorflow as tf 
from dense_net import dense_block

def cross_attention_layer(latent_dim, data_dim, projection_dim, dense_units):
    input_latent = layers.Input(shape=(latent_dim, projection_dim))
    input_data = layers.Input(shape=(data_dim, projection_dim))

    # Input first processed with a norm layer
    latent_array = layers.LayerNormalization()(input_latent)
    data_array = layers.LayerNormalization()(input_data)

    # QKV cross attention
    # K and V are projections of the input byte array, Q is a projection of a learned latent array
    q = layers.Dense(projection_dim)(latent_array)
    k = layers.Dense(projection_dim)(data_array)
    v = layers.Dense(projection_dim)(data_array)
    
    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention = layers.Attention(use_scale=True)(
        [q, k ,v]
    )
    
    # pass to a linear layer
    attention = layers.Dense(projection_dim) # ?
    # Add input to output
    attention = layers.Add()([attention, latent_array])

    # Normalize
    # attention = layers.LayerNormalization()(attention)

    # Pass the attention to a Dense block (MLPs)
    outputs = dense_block(dense_units)(attention)

    # Add input to output
    outputs = layers.Add()([outputs, attention])

    # Create the Keras model.
    model = tf.keras.Model(inputs=[input_latent, input_data], outputs=outputs)
    return model