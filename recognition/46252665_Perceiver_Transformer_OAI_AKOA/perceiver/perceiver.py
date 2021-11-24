"""
Perceiver Transformer Architecture

The perceiver mixes the latent self-attention mechanism with the cross-attention
mechanism. The input data only enters through the transformer through the
cross-attention mechanism. This allows the model to be of significant lower
size than the data array and solves the transformer quadratic compute bottleneck.

The perceiver transformer works for inputs such as images, videos, and
audio. No prior assumption or changes is required for any input modality,
for the model to work. Given an image, it will do transformer like attention
but since images are of large shape it is too much to put it in one transformer.
Therefore, it sub-divides the image into patches, and for each patch, it makes
a vector out of it. All the pixels are close together goes into one vector,
thus treated as a group.

The perceiver's goal is to have a low dimension latent array, N. The top row
is a regular self-attention transformer as in the original paper with the
quadratic blowup along with multiple transformer blocks. The image input to the
Perceiver is a byte array, M i.e. (228x228=51984), which is of higher dimension,
however the queries calculated are from the latent dimension rather than the
image itself. Thus resulting in a lower N x M blowup. Data of the image flows
into the cross-attention, generating a latent state which is transformed further
by the transformer and queries are generated to do cross-attention again to the
same image behaving like a recurrent neural network(RNN) with weight sharing
options. The data that flows through influences the queries and is refined
multiple times.

https://keras.io/examples/vision/perceiver_image_classification/
https://github.com/Rishit-dagli/Perceiver

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import tensorflow as tf
from perceiver.attention_module import TransformerModule
from perceiver.extract_patches import Patches
from perceiver.feed_forward import FeedForward
from perceiver.patch_encoder import PatchEncoder
from settings.config import *


class PerceiverTransformer(tf.keras.Model):
    """Perceiver architecture based on attention principles"""

    def __init__(self, data):
        super(PerceiverTransformer, self).__init__()

        self.data = data

        self.latent_array = None
        self.patcher = None
        self.patch_encoder = None
        self.cross_attention = None
        self.transformer = None
        self.global_average_pooling = None
        self.classification_head = None

    def build(self, input_shape):
        """builds the perceiver transformer module by initialising the
        following patches, patch_encoder, cross_attention,
        transformer and the classification_head."""
        # latent array
        self.latent_array = self.add_weight(
            shape=(LATENT_DIMENSION,
                   PROJECTION_DIMENSION), initializer="random_normal",
            trainable=True)

        # patching module
        self.patcher = Patches()

        # patch encoder
        self.patch_encoder = PatchEncoder()

        # cross-attention module
        self.cross_attention = TransformerModule().\
            create_cross_attention_module()

        # Transformer module
        self.transformer = TransformerModule().create_transformer_module()

        # global average pooling layer
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()

        # create a classification head
        self.classification_head = FeedForward().feed_forward_network()

        super(PerceiverTransformer, self).build(input_shape)

    def call(self, inputs):
        """Applies cross-attention and transformer in an alternative manner.

        The cross attention inputs are the latent_array of shape
        (LATENT_DIMENSION, PROJECTION_DIMENSION), data_array are the encoded
        patches which are positional embedding.

        Returns the final prediction"""
        # create patches
        patches = self.patcher(inputs)

        # encode patches
        encoded_patches = self.patch_encoder(patches)

        # cross-attention inputs
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }

        latent_array = None
        # apply the cross-attention and the Transformer modules
        for _ in range(ITERATIONS):
            # apply cross-attention from the latent array to the data array
            latent_array = self.cross_attention(cross_attention_inputs)

            # apply self-attention Transformer to the latent array
            latent_array = self.transformer(latent_array)

            # set the latent array of the next iteration
            cross_attention_inputs["latent_array"] = latent_array

        # apply global average pooling to generate a
        # (batch_size, projection_dim) representation tensor
        representation = self.global_average_pooling(latent_array)

        # return logits
        return self.classification_head(representation)
