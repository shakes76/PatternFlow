from tensorflow.keras import layers
import tensorflow as tf 
import copy

def transformer_layer(latent_size, proj_size, num_heads, num_trans_blocks):
    inputs_orig = layers.Input(shape=(latent_size, proj_size))

    input_plus_output = copy.deepcopy(inputs_orig)
    # Create multiple layers of the Transformer block.
    for _ in range(num_trans_blocks):
        # Layer norm
        norm = layers.LayerNormalization()(inputs_orig)
        # Create QKV self-attention layer.
        # Multihead becomes self-attetion when q = k = v. v = k if not supplied
        attention = layers.MultiHeadAttention(
            num_heads, proj_size)(norm, norm)

        # pass to a linear layer
        attention = layers.Dense(proj_size)(attention)

        # Add output to input
        attention = layers.Add()([attention, inputs_orig])

        # Apply layer normalizationn
        attention = layers.LayerNormalization()(attention)

        # Dense MLP block
        outputs = layers.Dense(proj_size, activation=tf.nn.gelu)(attention)

        # Final linear layer
        outputs = layers.Dense(proj_size)(outputs)

        # Skip connection 2.
        input_plus_output = layers.Add()([outputs, attention])

    # Create the Keras model.
    return tf.keras.Model(inputs=inputs_orig, outputs=input_plus_output)