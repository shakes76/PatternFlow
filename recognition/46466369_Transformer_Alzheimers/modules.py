import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
import dataset
import train
import predict

"""
modules.py
Contains the source code of the components of the model. Each component is implemented as a class or function.
"""

"""
Patch function
Splits the image into patches of dimension (p_size, p_size).
This is implemented as a layer
"""
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(tf.convert_to_tensor([images]))[0]

        # splits the image into patches of dimension patch_size x patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

"""
Multilayer perceptron (MLP)
Multiple (n = layer_count) dense layers with dropout
"""
def MLP(layer, layer_count, dropout):
    for count in layer_count:
        layer = layers.Dense(count, activation='GeLU')(layer)
        layer = layers.Dropout(dropout)(layer)
    return layer


class PatchEncoder(layers.Layer):
  def __init__(self, num_patches, projection_dim):
    super(PatchEncoder, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )

  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded
