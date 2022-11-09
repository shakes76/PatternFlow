"""
This file contains the perceiver model, based on the
tutorial for building a perciever on the cifar10 dataset:
https://keras.io/examples/vision/perceiver_image_classification/
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dense_network(dense_units, dropout):
    """
    Returns a dense network of data projection size units.
    :param dense_units: a two item list describing the dimensions of the dense network.
    :param dropout: the dropout rate to use in the drop out layer
    :return:
    """

    # create dense block
    dense_layers = [layers.Dense(units, activation=tf.nn.gelu) for units in dense_units[:-1]]
    dense_layers.append(layers.Dense(units=dense_units[-1]))
    dense_layers.append(layers.Dropout(dropout))

    return keras.Sequential(dense_layers)


class Patches(layers.Layer):
    """
    Layer which creates patches from an input set of images.
    """
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


class PatchEncoder(layers.Layer):
    """
    Layer for encoding image patches of the data array into the latent vector.
    """

    def __init__(self, num_patches, projection_size):

        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches

        # projection dense layer and embedding layer
        self.projection = layers.Dense(units=projection_size)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_size
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


def cross_attention_module(
        latent_array_size,
        data_array_size,
        dense_units,
        dropout
):
    """
    Builds and returns the cross-attention module of the model (i.e. correlation
    between latent array, image data array and each class).

    :param latent_array_size: the size of the latent encoding array
    :param data_array_size: the size of the patched image data array
    :param latent_array_size: the size to project the data array to the latent array
    :param dense_units: the number of dense units in the dense block
    :param dropout: the rate of dropout in the dense block
    :return:
    """

    inputs = {
        # latent array [1, latent_dim, latent_array_size].
        "latent_array": layers.Input(shape=(latent_array_size, latent_array_size)),
        # data_array (encoded image) [batch_size, data_dim, latent_array_size].
        "data_array": layers.Input(shape=(data_array_size, latent_array_size)),
    }

    # normalise latent array and data array
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Q array: [1, latent_dim, latent_array_size].
    Q = layers.Dense(units=latent_array_size)(latent_array)

    # K array: [batch_size, data_dim, latent_array_size].
    K = layers.Dense(units=latent_array_size)(data_array)

    # V array: [batch_size, data_dim, latent_array_size].
    V = layers.Dense(units=latent_array_size)(data_array)

    # cross-attention outputs: [batch_size, latent_dim, latent_array_size].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [Q, K, V], return_attention_scores=False
    )

    # first skip connection with normalisation
    attention_output = layers.Add()([attention_output, latent_array])
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

    # create and apply dense layers
    dense = dense_network(dense_units, dropout)
    outputs = dense(attention_output)

    # second skip connection
    outputs = layers.Add()([outputs, attention_output])

    # return model
    return keras.Model(inputs=inputs, outputs=outputs)


def latent_transformer_module(
    latent_array_size,
    num_heads,
    transformer_blocks,
    dense_units,
    dropout,
):
    """
    Creates a transformer module for self attention of the latent array over
    a number of blocks.

    :param latent_array_size: the size of the latent array for self-attention
    :param num_heads: the number of multihead attention heads for self-attention
    :param transformer_blocks: the number of iterations of transformer blocks
    :param dense_units: the size of the dense network in each transformer block
    :param dropout: the rate of dropout in the transformer dense network
    :return:
    """

    # input_shape: [1, latent_dim, latent_array_size]
    inputs = layers.Input(shape=(latent_array_size, latent_array_size))
    outputs = inputs

    # iteratively generate transformer block
    for i in range(transformer_blocks):

        # normalise
        norm_one = layers.LayerNormalization(epsilon=1e-6)(outputs)

        # multi-head self-attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=latent_array_size, dropout=0.1
        )(norm_one, norm_one)

        # first skip connection
        skip_one = layers.Add()([attention_output, outputs])

        # normalise again and apply dense network
        norm_two = layers.LayerNormalization(epsilon=1e-6)(skip_one)
        dense = dense_network(dense_units, dropout)(norm_two)

        # second skip connection
        outputs = layers.Add()([dense, skip_one])

    # return model
    return keras.Model(inputs=inputs, outputs=outputs)


class Perceiver(keras.Model):
    """
    Class representing full model of the perceiver transformer.
    Default settings have been optimised for classifying laterality
    in AKOA dataset.
    """

    def __init__(
        self,
        input_img_size,
        num_classes,
        patch_size: int = 2,
        latent_array_size: int = 128,
        transformer_heads: int = 8,
        transformer_blocks: int = 4,
        dropout: int = 0.2,
        model_iterations: int = 2,
    ):
        super(Perceiver, self).__init__()

        # size of  latent array
        self.latent_array_size = latent_array_size

        # size of patch-encoded image data array
        self.data_array_size = (input_img_size // patch_size) ** 2

        # size of patches to be extracted from train images
        self.patch_size = patch_size

        # number of transformer heads for self-attention
        self.transformer_heads = transformer_heads

        # number of blocks of transformers in each model iteration
        self.transformer_blocks = transformer_blocks

        # size of the transformer dense network
        self.dense_units = (latent_array_size, latent_array_size)
        self.dropout = dropout

        # repetitions of cross-attention/transformer modules
        self.model_iterations = model_iterations

        # size of dense network of the final classifier
        self.classifier_units = (latent_array_size, num_classes)

        print(f"Image size: {input_img_size} X {input_img_size} = {input_img_size ** 2}")
        print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
        print(f"Elements per patch: {(patch_size ** 2)}")
        print(f"Latent array shape: {latent_array_size} X {latent_array_size}")

    def build(self, input_shape):

        # latent array
        self.latent_array = self.add_weight(
            shape=(self.latent_array_size, self.latent_array_size),
            initializer="random_normal",
            trainable=True,
        )

        # patching module
        self.patcher = Patches(self.patch_size)

        # patch encoder
        self.patch_encoder = PatchEncoder(self.data_array_size, self.latent_array_size)

        # cross-attention module
        self.cross_attention = cross_attention_module(
            self.latent_array_size,
            self.data_array_size,
            self.dense_units,
            self.dropout,
        )

        # transformer module.
        self.transformer = latent_transformer_module(
            self.latent_array_size,
            self.transformer_heads,
            self.transformer_blocks,
            self.dense_units,
            self.dropout,
        )

        # global average pooling
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # dense classification block
        self.classification_head = dense_network(self.classifier_units, self.dropout)

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):

        # create patches
        patches = self.patcher(inputs)

        # encode patches
        encoded_patches = self.patch_encoder(patches)

        # cross-attention inputs
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }

        # iteratively apply cross-attention and transformer modules
        for i in range(self.model_iterations):

            # apply cross attention between latent and data array
            latent_array = self.cross_attention(cross_attention_inputs)

            # apply self-attention on latent array in transformer module
            latent_array = self.transformer(latent_array)

            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # global average pooling to generate a [batch_size, latent_array_size] representation
        representation = self.global_average_pooling(latent_array)

        # get logits from final classification layers
        logits = self.classification_head(representation)
        return logits


