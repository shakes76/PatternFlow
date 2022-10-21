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
def MLP(layer, layer_counts, dropout):
    for count in layer_counts:
        layer = layers.Dense(count, activation=tf.keras.activations.gelu)(layer)
        layer = layers.Dropout(dropout)(layer)
    return layer

"""
A patch encoder is required to retain the image order information. We map the flattened patches to projection_dim through a dense layer.
In addition, we embed the position of the patch in the image with an embedding layer.
"""
class PatchEmbed(layers.Layer):
  def __init__(self, num_patches, projection_dim):
    super(PatchEmbed, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )

  def call(self, patch):
    # store positions as integer values of the all the patches
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    # project the patch and add the position to it
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded

def create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, num_classes, num_heads, transformer_layers, mlp_layer_counts):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches with their embedded position
    embedded = PatchEmbed(num_patches, projection_dim)(patches)

    """
    Feed transformer encoder from ViT paper:
    embedded patches -> norm (layer) -> multi-head attention --(+ encoded)--> norm (layer) -> mlp
    Classify with softmax:
    mlp --(+ norm2)--> output(softmax)
    """
    for _ in range(transformer_layers):
        norm1 = layers.LayerNormalization(epsilon=0.001)(embedded)
        # multi_attention needs 2D input since we are applying it to 2D data.
        multi_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(norm1, norm1)
        addembed = layers.Add()([multi_attention, embedded])
        norm2 = layers.LayerNormalization(epsilon=0.001)(addembed)
        mlp = MLP(norm2, mlp_layer_counts, dropout=0.5)
        addnorm = layers.Add()([mlp, norm2])

    output = layers.Dense(num_classes, activation="softmax")(addnorm)
    model = keras.Model(inputs=inputs, outputs=output)

    return model