"""
Transformer Attention Module

https://keras.io/examples/vision/perceiver_image_classification/
https://github.com/Rishit-dagli/Perceiver

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import tensorflow as tf

from perceiver.feed_forward import FeedForward
from settings.config import *


class TransformerModule:
    """Attention Module provides the cross attention and multi headed attention
    as per the transformer blocks."""

    @staticmethod
    def apply_layer_normalisation(_to):
        """Apply later normalisation to argument"""
        return tf.keras.layers.LayerNormalization(epsilon=EPSILON)(_to)

    @staticmethod
    def skip_connection(_from, _to):
        """Apply skip connection _from to _to argument

        Skip Connections: connection between nodes in different layers skipping
        one or more layers of nonlinear processing.
        """
        return tf.keras.layers.Add()([_from, _to])

    @staticmethod
    def create_dense_layer(_to):
        """Create dense layer of projected dimension to input argument"""
        return tf.keras.layers.Dense(units=PROJECTION_DIMENSION)(_to)

    @staticmethod
    def get_attention(x1=None, query=None, key=None, value=None,
                      attention=False):
        """Returns keras attention if attention argument is true and
        query, key, value is provided else returns multi headed attention
        for argument x1"""
        return tf.keras.layers.Attention(use_scale=True,
                                         dropout=DROPOUT)(
            [query, key, value], return_attention_scores=False) if attention \
            else tf.keras.layers.MultiHeadAttention(
            num_heads=TRANSFORMER_HEADS,
            key_dim=PROJECTION_DIMENSION,
            dropout=DROPOUT)(x1, x1)

    def create_cross_attention_module(self):
        """Attention performed on queries generated from embedding and
        keys, values generated from another embeddings.

        The query vectors, key vectors, and value vectors are the three types of
        vectors calculated in the transformer architecture. These are determined
        by multiplying the input with a linear transformation. Each input is the
        same in self attention, however they can be different in cross
        attention. The purpose of cross attention is to calculate attention
        scores utilising data from different sources. For instance, the query
        vector in the perceiver transformer is calculated from the latent array,
        while the key and value is generated from the image data.

        Input is a dictionary containing keys 'latent_array' and
        'data_array'. latent_array is an input of latent dimension and
        PROJECTION_DIMENSION whereas the data array is of shape patches and
        PROJECTION_DIMENSION.

        We apply layer normalisation to both these inputs. Query is created
        from the The latent_array by applying dense layer, and
        key, value is created by applying the same for the data_array which
        are generated from the encoded image.

        Layer normalisation is applied to attention and feed to the sequential
        neural net for vector of raw predictions, to produce another
        (LATENT_DIMENSION, PROJECTION_DIMENSION).

        Returns Keras Model
        """
        inputs = {
            "latent_array": tf.keras.layers.Input(
                shape=(LATENT_DIMENSION, PROJECTION_DIMENSION)),
            "data_array": tf.keras.layers.Input(
                shape=(PATCHES, PROJECTION_DIMENSION)),
        }

        # apply layer normalization
        latent_array = self.apply_layer_normalisation(inputs["latent_array"])
        data_array = self.apply_layer_normalisation(inputs["data_array"])

        # query tensor: (1, latent_dim, projection_dim)
        query = self.create_dense_layer(latent_array)

        # key tensor: (data_dim, projection_dim)
        key = self.create_dense_layer(data_array)

        # value tensor: (data_dim, projection_dim)
        value = self.create_dense_layer(data_array)

        # cross-attention outputs: (latent_dim, projection_dim)
        attention_output = self.get_attention(
            x1=None,
            query=query,
            key=key,
            value=value,
            attention=True
        )

        # skip connection
        attention_output = self.skip_connection(attention_output, latent_array)

        # apply layer normalization
        attention_output = self.apply_layer_normalisation(attention_output)

        # apply sequential feed forward network
        ffn = FeedForward().feed_forward_network()
        outputs = ffn(attention_output)

        # skip connection
        outputs = self.skip_connection(outputs, attention_output)

        # return keras model
        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='Cross_Attention'
        )

    def create_transformer_module(self):
        """
        The transformer block in perceiver uses GPT2 architecture which is based
        on the decoder of the original transformer. The decoder takes that input
        representation from the cross attention and, the self-attention layer
        helps the decoder focus on appropriate places in the input sequence.
        Self-attention is the result if the query, key, and value are all the
        same. In each time-step in the query corresponds to a sequence in the
        key, and the result is a fixed-width vector. The query and key tensors
        are then scaled and dot-produced.

        Applies MultiHeadAttention to the cross attention module to
        LATENT_DIMENSION. Iterations are applied multiple times as the number of
        TRANSFORMER_BLOCKS for MultiHeadAttention. Followed by the sequential
        neural net for vector of raw predictions, to produce another
        (LATENT_DIMENSION, PROJECTION_DIMENSION).

        Returns Keras Model
        """
        inputs = tf.keras.layers.Input(
            shape=(LATENT_DIMENSION, PROJECTION_DIMENSION))

        x0 = inputs
        # create multiple layers of the Transformer block
        for _ in range(TRANSFORMER_BLOCKS):
            # apply layer normalization
            x1 = self.apply_layer_normalisation(x0)

            # multi-head self-attention layer
            attention_output = self.get_attention(
                x1=x1,
                query=None,
                key=None,
                value=None,
                attention=False
            )

            # skip connection
            x2 = self.skip_connection(attention_output, x0)

            # apply layer normalization
            x3 = self.apply_layer_normalisation(x2)

            # apply sequential feed forward network
            ffn = FeedForward().feed_forward_network()

            x3 = ffn(x3)

            # skip connection
            x0 = self.skip_connection(x3, x2)

        # return keras model
        return tf.keras.Model(
            inputs=inputs,
            outputs=x0,
            name='Latent_Transformer'
        )
