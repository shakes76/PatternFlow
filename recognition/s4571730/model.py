import tensorflow as tf
import math
import numpy as np
from tensorflow.keras import layers
import copy

# def fourier_transform(img, bands, sampling_rate):
#     # data has 2 dimensions 
#     num_row, num_col, _ = img.shape
#     encodings = []
#     x_row = [(idx // num_col)/ (num_row - 1) * 2 - 1 for idx in list(range(num_row*num_col))] # row, col in range -1 1
#     x_col = [(idx % num_col)/ (num_col - 1) * 2 - 1 for idx in list(range(num_row*num_col))]
#     for input in range(num_col*num_row):
#         encoding = []
#         for xd in [x_row[input], x_col[input]]:
#             freq = np.logspace(0.0, math.log(sampling_rate/2) / math.log(10), bands, dtype=np.float32)
#             encoded_concat = []
#             for i in range(bands):
#                 encoded_concat.append(math.sin(freq[i] * math.pi * xd))
#                 encoded_concat.append(math.cos(freq[i] * math.pi * xd))
#             encoded_concat.append(xd)
#             encoding.extend(encoded_concat)
#         encodings.append(encoding)
#     return encodings

def fourier_encode(x, max_freq=10, num_bands=4):
    rows = x.shape[0]
    cols = x.shape[1]
    xr = tf.linspace(-1, 1, rows)
    xc = tf.linspace(-1, 1, cols)
    xd = tf.reshape(tf.stack(tf.reverse(tf.meshgrid(xr, xc), axis=[-3]),axis=2), (rows, cols, 2))
    xd = tf.repeat(tf.expand_dims(xd, -1), repeats=[2*bands + 1], axis=3)

    freq = tf.experimental.numpy.logspace()
    # x = tf.expand_dims(x, -1)
    # x = tf.cast(x, dtype=tf.float32)
    # orig_x = x

    # scales = tf.reshape(tf.experimental.numpy.logspace(
    #     1.0,
    #     tf.math.log(max_freq / 2) / math.log(10),
    #     num=num_bands,
    #     dtype=tf.float32,
    # ), (1,1,1,2 * max_freq - 1) )
    # scales *= math.pi
    # x = x * scales 
    # return tf.concat((tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1), orig_x), axis=-1)
    # return x

def cross_attention(latent_dim, data_dim, projection_dim, dense_units):
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
    return


def dense_block(dense_layers):
    block = tf.keras.Sequential()
    for units in dense_layers[:-1]:
        # use GELU activation as in paper
        block.add(layers.LayerNormalization())
        block.add(layers.Dense(units, activation=tf.nn.gelu))

    # Final linear layer
    block.add(layers.Dense(units=dense_layers[-1]))
    # block.add(layers.Dropout(dropout_rate))
    return block

    

def transformer(latent_dim, projection_dim, num_heads, num_transformer_blocks, dense_layers):
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    input_orig = copy.deepcopy(inputs)
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformer_blocks):

        # Layer norm
        norm = layers.LayerNormalization()(inputs)
        # Create QKV self-attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads, projection_dim)(norm, norm)

        # pass to a linear layer
        attention_output = layers.Dense(projection_dim) # ?

        # Add output to input
        attention_output = layers.Add()([attention_output, input_orig])

        # Apply layer normalization 2.
        # x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # Dense MLP block
        output = dense_block(dense_layers)(attention_output)

        # Skip connection 2.
        final_output = layers.Add()([output, attention_output])

    # Create the Keras model.
    return tf.keras.Model(inputs=inputs, outputs=final_output)




