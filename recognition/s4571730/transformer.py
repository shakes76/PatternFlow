from tensorflow.keras import layers
import tensorflow as tf 
import copy
from dense_net import dense_block

def transformer_layer(latent_dim, projection_dim, num_heads, num_transformer_blocks, dense_layers):
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