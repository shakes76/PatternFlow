from tensorflow.keras import layers
import tensorflow as tf 
import copy

"""
Create a self-attention (transformer) layer, with structure mirroring the paper's spec
"""
class Transformer(layers.Layer):
    """
    The transformer layer. Process params and create a model.

    Params:
        latent_size: int, size of the latent dimension
        proj_size: int, embedding size of fourier features, applied to
                        each element in the data and latent arrays
        num_heads: int number of heads in the MultiHeadAttention layer
        num_trans_blocks: int, number of transformer blocks in the model
    """
    def __init__(self, proj_size, num_heads, num_trans_blocks):
        super(Transformer, self).__init__()
        self.norm = layers.LayerNormalization()
        self.attention = layers.MultiHeadAttention(num_heads, proj_size)
        self.dense = layers.Dense(proj_size)
        self.add = layers.Add()
        self.dense_gelu = layers.Dense(proj_size, activation=tf.nn.gelu)
        self.num_trans_blocks = num_trans_blocks

    """
    Call the transformer layer to model the input imgs

    Params:
        inputs: output from the cross-attention layer

    Returns: 
        QKV self-attention
    """
    def call(self, inputs):
        # Create multiple layers of the Transformer block.
        input_plus_output = inputs
        for _ in range(self.num_trans_blocks):
            # Layer norm
            norm = self.norm(inputs)
            # Create QKV self-attention layer.
            # Multihead becomes self-attetion when q = k = v. v = k if not supplied
            attention = self.attention(norm, norm)

            # pass to a linear layer
            attention = self.dense(attention)

            # Add output to input
            attention = self.add([attention, inputs])

            # Apply layer normalizationn
            attention = self.norm(attention)

            # Dense MLP block
            outputs = self.dense_gelu(attention)

            # Final linear layer
            outputs = self.dense(outputs)

            # Add output to input
            input_plus_output = self.add([outputs, attention])

        return input_plus_output
