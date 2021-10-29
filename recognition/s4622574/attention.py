from tensorflow.keras import layers
import tensorflow as tf 

def cross_attention_layer(latent_size, data_size, proj_size):

    input_latent = layers.Input((latent_size, proj_size))
    latent_array = layers.LayerNormalization()(input_latent)

    input_data = layers.Input((data_size, proj_size))
    data_array = layers.LayerNormalization()(input_data)

    q = layers.Dense(proj_size)(latent_array)
    k = layers.Dense(proj_size)(data_array)
    v = layers.Dense(proj_size)(data_array)
    
    attention = layers.Attention(use_scale=True)([q, k ,v])
    
    attention = layers.Dense(proj_size)(attention)

    attention = layers.Add()([attention, latent_array])


    attention = layers.LayerNormalization()(attention)

    outputs = layers.Dense(proj_size, activation=tf.nn.gelu)(attention)

    outputs = layers.Dense(proj_size)(outputs)

    outputs = layers.Add()([outputs, attention])

    return tf.keras.Model(inputs=[input_latent, input_data], outputs=outputs)