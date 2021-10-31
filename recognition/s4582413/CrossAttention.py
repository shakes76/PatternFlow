from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

"""
The cross attention layer for the perceiver transformer as described in the paper
    latent_size = 256
    data_size = 73 * 64
    proj_size = 2 * (2*num_bands + 1) + 1, with num_bands being the number of bands in the Fourier encode
    img_size = size of image data
"""
def cross_attention(img_size):

    # latent array for the model
    latent_array = layers.Input \
        (shape=(256, 2* (2 * 6 + 1) + 1))  # latent array used in the paper, with latent size =256,
    # proj_size = 2 * (2*num_bands + 1) + 1, with num_bands being the number of bands in the Fourier encode (6 here)

    # data input array for the model
    input_array = layers.Input(shape=(img_size, 2*(2*6 + 1) + 1))

    # apply normalisation
    latent_array = layers.LayerNormalization(epsilon=1e-6)(latent_array)
    input_array = layers.LayerNormalization(epsilon=1e-6)(input_array)

    # Now obtaining the query, key and value vector
    # get query, then key and value vector
    qkv = []

    query = layers.Dense(units=2*(2*6 + 1) + 1)(latent_array)
    key = layers.Dense(units=2*(2*6 + 1) + 1)(input_array)
    value = layers.Dense(units=2*(2*6 + 1) + 1)(input_array)

    qkv.append(query)
    qkv.append(key)
    qkv.append(value)

    # apply attention transformation as described in paper
    attention = layers.Attention(use_scale=True, dropout=0.1)(
        qkv, return_attention_scores=False
    )

    # Adding input to output
    input_output = [attention, latent_array]
    attention = layers.Add()(input_output)
    # normalising the attention layer
    attention = layers.LayerNormalization(epsilon=1e-6)(attention)

    # Now pass the attention result to linear layers
    mlp = []
    linear_layer = layers.Dense(units=2*(2*6 + 1) + 1)
    mlp.append(linear_layer)
    mlp_output = keras.Sequential(mlp)(attention)
    # Again, adding input to output
    mlp_output = layers.Add()([mlp_output, attention])

    # the latent input for the model
    latent_input =  layers.Input(shape=(256, 2* (2 * 6 + 1) + 1))
    # the data input for the model
    data_input = layers.Input(shape=(img_size, 2 * (2 * 6 + 1) + 1))
    
    # create the corresponding keras model and returns it
    return keras.Model(inputs=[latent_input, data_input], outputs=mlp_output)