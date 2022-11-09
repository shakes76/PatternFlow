"""
    Perceiver implemented with reference to the ViT Paper 
    and https://github.com/google-research/vision_transformer 
    as in JAX. 

    The Perceiver paper layer image was referenced as well as the keras 
    basic example code. 
    https://keras.io/examples/vision/perceiver_image_classification/

    The patch embeddings are just linear numbers from 0 to N, 
    increasing from row to row. 
    e.g. 
    0 1 2
    3 4 5
    6 7 8  (if image_size // patch_size == 3)

    In the paper, Fourier encodings are used.
    However in this code I could not get them to work.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow_addons as tfa
import math
import data as d
import config as c


def latent_transformer_encoder():
    """
        the latent transformer takes input 
        by the latent array given by the Cross-Attention 
        layer 

        based on vit_figure.png available 
        https://github.com/google-research/vision_transformer

        as described the transformer in the VIT paper has the 
            following architecture

        Input: embedded patches-----|
            |                       |
            |                       |
            Layer Normalization     |
            |                       |
            |                       |
            Multi-Head Attention    |
            |                       |
            |                       |
            Add-----------<--------<|
            |
            |-->------->------->----|
            |                       |
            Layer Normalization     |
            |                       |
            |                       |
            MLP                     |
            |                       |
            Add----<------<-----<---|
            |
        Output: patches
    """
    # the input takes in the latent array
    inputs = layers.Input(shape=(c.latent_dim, c.projection_dim))
    x0 = inputs 

    # Transformer Encoder as described in ViT
    for block in range(c.num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # get the attn output
        attn_out = layers.MultiHeadAttention(
            num_heads=c.num_heads, key_dim=c.projection_dim, dropout=0.1
        )(x1, x1)
        #skip connection from x0, and add with Multi-Head Attention output
        x2 = layers.Add()([attn_out, x0])
        # Normalization layer
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        mlp = keras.Sequential([
            layers.Dense(256, activation=tf.nn.gelu),
            layers.Dense(256),
            layers.Dropout(c.dropout_rate)
        ])
        #mlp call
        x3 = mlp(x3)
        # add the output of the mlp and the skip connection from the output of last Add()
        x0 = layers.Add()([x3, x2])

    model = keras.Model(inputs=inputs, outputs=x0)
    return model


def cross_attention():
    """
        The cross attention module takes in the latent array twice, first as 
        itself, then as the query tensor.

        The module also takes in the image data as keys and value tensors. 
        
        The module ouputs the latent array for the next latent transformer.
    """
    # Receive the input tensors as an array.
    #0 -> latents
    #1 -> data
    inputs = [
        layers.Input(shape=(c.latent_dim, c.projection_dim)),
        layers.Input(shape=(c.num_patches, c.projection_dim))
    ]

    #apply normalization to inputs
    latent_tensor = layers.LayerNormalization(epsilon=1e-6)(inputs[0])
    data_tensor = layers.LayerNormalization(epsilon=1e-6)(inputs[1])

    # densely connected layers map the latent_tensor and data_tensor 
    #   to the queries, keys and values for the attention layer.
    query = layers.Dense(c.projection_dim)(latent_tensor)
    key = layers.Dense(c.projection_dim)(data_tensor)
    value = layers.Dense(c.projection_dim)(data_tensor)

    # The cross attention layer takes in the queries, keys 
    #   and values to output
    attention = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False 
    )


    attention = layers.Add()([attention, latent_tensor])

    attention = layers.LayerNormalization(epsilon=1e-6)(attention)

    mlp = keras.Sequential([
        layers.Dense(256, activation=tf.nn.gelu),
        layers.Dense(256),
        layers.Dropout(c.dropout_rate)
    ])

    outputs = mlp(attention)

    outptus = layers.Add()([outputs, attention])

    return keras.Model(inputs=inputs, outputs=outputs)

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # batch_size = 
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

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


class Perceiver(keras.Model):
    def __init__(self):
        super(Perceiver, self).__init__()

    def build(self, input_shape):
        self.latent_array = self.add_weight(
            shape=(c.latent_dim, c.projection_dim),
            initializer="random_normal",
            trainable=True
        )

        self.patcher = Patches(c.patch_size)

        self.patch_encoder = PatchEncoder(c.num_patches, c.projection_dim)

        self.cross_attention = cross_attention()

        self.latent_transformer = latent_transformer_encoder()

        self.pooling = layers.GlobalAveragePooling1D() 

        self.classification_head = keras.Sequential([
            layers.Dense(c.projection_dim, activation=tf.nn.gelu),
            layers.Dense(c.num_classes),
            layers.Dropout(c.dropout_rate)
        ])

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        patches = self.patcher(inputs)
        encoded_patches = self.patch_encoder(patches)

        cross_attn_input = [
            tf.expand_dims(self.latent_array, 0),
            encoded_patches
        ]

        for _ in range(c.num_iterations):
            latent_array = self.cross_attention(cross_attn_input)
            latent_array = self.latent_transformer(latent_array)
            cross_attn_input[0] = latent_array 

        representation = self.pooling(latent_array)
        return self.classification_head(representation)