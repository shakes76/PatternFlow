from tensorflow.keras import layers
import tensorflow as tf 

def attention_mechanism(latent_size, data_size, proj_size):

    input_latent = layers.Input((latent_size, proj_size))
    latents = layers.LayerNormalization()(input_latent)

    input_data = layers.Input((data_size, proj_size))
    data_array = layers.LayerNormalization()(input_data)

    query = layers.Dense(proj_size)(latents)
    key = layers.Dense(proj_size)(data_array)
    value = layers.Dense(proj_size)(data_array)
    
    attention = layers.Attention(use_scale=True)([query, key ,value])
    
    attention = layers.Dense(proj_size)(attention)

    attention = layers.Add()([attention, latents])

    attention = layers.LayerNormalization()(attention)

    outputs = layers.Dense(proj_size, activation=tf.nn.gelu)(attention)

    outputs = layers.Dense(proj_size)(outputs)

    outputs = layers.Add()([outputs, attention])

    return tf.keras.Model(inputs=[input_latent, input_data], outputs=outputs)