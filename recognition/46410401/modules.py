import keras.backend
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
from numba import cuda
import gc

# Set parameters of the model
num_classes = 2
input_shape = (256, 240, 3)
weight_decay = 0.0001
batch_size = 64
image_size = 72  # Image will be resized this size
patch_size = 4  # Number of patches, Image size must be divisible by this number
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4  # Number of heads in multi-head attention
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8  # Depth of the model
mlp_head_units = [2048, 1024]
transformer_dropout = 0.1
mlp_dropout = 0.5
epsilon = 1e-6


# Multi-layer perceptron
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
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


data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)


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


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    # Patches Created
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create Transformer Layers.
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=epsilon)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=transformer_dropout
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=epsilon)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=transformer_dropout)
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=epsilon)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(mlp_dropout)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=mlp_dropout)
    # Classify final outputs
    logits = layers.Dense(num_classes)(features)
    # Create the model
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# VIT with hypertuning support. May cause GPU OOM on differing architectures
def create_tuned_classifier(hp):
    gc.collect()

    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)

    # Patches Created
    hp_patch_size = hp.Choice('patch_size', values=[2, 4, 8])
    patches = Patches(patch_size)(augmented)

    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    hp_transformers = hp.Choice('transformers', values=[2, 4, 8])
    hp_dropout = hp.Choice('dropout', values=[0.1, 0.2, 0.3])

    # Create Transformer Layers.
    for _ in range(hp_transformers):
        x1 = layers.LayerNormalization(epsilon=epsilon)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=hp_dropout
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=epsilon)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=hp_dropout)
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=epsilon)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(mlp_dropout)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=mlp_dropout)

    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    optimizer = tfa.optimizers.AdamW(
        learning_rate=hp_learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model
