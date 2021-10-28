import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Perceiver:
    """
    Creates and returns several dense layers followed by a dropout layer
    The number of neurons within each dense layer is specified in an array of integers
    """
    @staticmethod
    def create_dense_layers(output_units, dropout_rate):
        dense_layers = []
        # Loop through, adding each dense layer
        for units in output_units[:-1]:
            dense_layers.append(layers.Dense(units, activation=tf.nn.gelu))
        dense_layers.append(layers.Dense(units=output_units[-1]))
        dense_layers.append(layers.Dropout(dropout_rate))
        dense = keras.Sequential(dense_layers)
        return dense

    """
    Creates and returns a cross attention model which forms a core module of the perceiver.
    The input passes through an attention layer which is added to the latent array input and then normalised.
    This is then passed through several dense layers and the output of this is then added to the attention output
    which is fed in through a skip connection. 
    """
    @staticmethod
    def create_cross_attention_module(latent_size, data_dim, projection_dim, dense_units, dropout_rate):
        inputs = {
            "latent_array": layers.Input(shape=(latent_size, projection_dim)),
            "data_array": layers.Input(shape=(data_dim, projection_dim)),
        }

        # Normalise the latent and data inputs independently
        latent_array = layers.LayerNormalization()(inputs["latent_array"])
        data_array = layers.LayerNormalization()(inputs["data_array"])

        # Create query, key and value vectors through dense layers
        query = layers.Dense(units=projection_dim)(latent_array)
        key = layers.Dense(units=projection_dim)(data_array)
        value = layers.Dense(units=projection_dim)(data_array)

        # Generate cross-attention outputs.
        attention_output = layers.Attention(use_scale=True, dropout=0.1)([query, key, value], return_attention_scores=False)
        # Sum the Latent array obtained through a skip connection to the attention output
        attention_output = layers.Add()([attention_output, latent_array])
        # Apply layer norm.
        attention_output = layers.LayerNormalization()(attention_output)

        # Apply dense layers
        ffn = Perceiver.create_dense_layers(dense_units, dropout_rate)
        outputs = ffn(attention_output)
        # Sum the dense output with the attention output using a skip connection
        outputs = layers.Add()([outputs, attention_output])

        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    """
    """
    @staticmethod
    def create_transformer_module(latent_size, projection_dim, num_heads, transformer_layers, dense_units, dropout_rate):
        return

