from tensorflow.keras import layers
import tensorflow as tf 
import copy

def transformer_layer(latent_size, proj_size, num_heads, num_trans_blocks):
    inputs_orig = layers.Input(shape=(latent_size, proj_size))

    input_plus_output = copy.deepcopy(inputs_orig)

    for _ in range(num_trans_blocks):

        norm = layers.LayerNormalization()(inputs_orig)


        attention = layers.MultiHeadAttention(
            num_heads, proj_size)(norm, norm)


        attention = layers.Dense(proj_size)(attention)


        attention = layers.Add()([attention, inputs_orig])


        attention = layers.LayerNormalization()(attention)


        outputs = layers.Dense(proj_size, activation=tf.nn.gelu)(attention)


        outputs = layers.Dense(proj_size)(outputs)


        input_plus_output = layers.Add()([outputs, attention])


    return tf.keras.Model(inputs=inputs_orig, outputs=input_plus_output)