"""
Creates positional embeddings for input data

https://keras.io/examples/vision/perceiver_image_classification/
https://github.com/Rishit-dagli/Perceiver

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""


import tensorflow as tf

from settings.config import *


class PatchEncoder(tf.keras.layers.Layer):
    """Projects patches on a latent dimension, 256, and linearly transforms
    and learns a positional embedding to the projection.

    Positional Encoding are generally applicable to all input modality. The
    original paper implemented the fourier feature positional embedding. The
    position used here is used to encode the relationship between the nearby
    patches.

    * The feature based embeddings allows the network to learn the position
        structure.
    * Produces sufficient results compared to ImageNet without prior
        assumptions.
    * Can be extended to multi-modal data.
    """

    def __init__(self):
        """Patch Encoder constructor method initialises the
        num_patches: PATCHES,
        projection: PROJECTION_DIMENSION,
        the positional_embedding of input_dim PATCHES,
        and output_dim PROJECTION_DIMENSION"""
        super(PatchEncoder, self).__init__()

        self.num_patches = PATCHES
        self.projection = tf.keras.layers.Dense(units=PROJECTION_DIMENSION)
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=PATCHES,
            output_dim=PROJECTION_DIMENSION
        )

    def call(self, patches):
        """Returns the positional embedding vector"""
        return self.projection(patches) + self.positional_embedding(tf.range(
            start=0, limit=self.num_patches, delta=1))
