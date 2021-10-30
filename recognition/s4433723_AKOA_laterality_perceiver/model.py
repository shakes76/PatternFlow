"""
REFERENCE FOR PERCEIVER HELP ON CIFAR10
https://keras.io/examples/vision/perceiver_image_classification/
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dense_network(dense_units, dropout):
    """

    :param dense_units:
    :param dropout:
    :return:
    """

    # create dense block
    dense_layers = [layers.Dense(units, activation=tf.nn.gelu) for units in dense_units[:-1]]
    dense_layers.append(layers.Dense(units=dense_units[-1]))
    dense_layers.append(layers.Dropout(dropout))

    return keras.Sequential(dense_layers)


class Patches(layers.Layer):
    """
    Class which creates patches from an input set of images.
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


def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, dense_units, dropout_rate
):

    inputs = {
        # Recieve the latent array as an input of shape [1, latent_dim, projection_dim].
        "latent_array": layers.Input(shape=(latent_dim, projection_dim)),
        # Recieve the data_array (encoded image) as an input of shape [batch_size, data_dim, projection_dim].
        "data_array": layers.Input(shape=(data_dim, projection_dim)),
    }

    # Apply layer norm to the inputs
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Create Q array: [1, latent_dim, projection_dim].
    Q = layers.Dense(units=projection_dim)(latent_array)

    # Create K array: [batch_size, data_dim, projection_dim].
    K = layers.Dense(units=projection_dim)(data_array)

    # Create V array: [batch_size, data_dim, projection_dim].
    V = layers.Dense(units=projection_dim)(data_array)

    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [Q, K, V], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

    # Dense layers
    dense = dense_network(dense_units, dropout_rate)

    # Apply dense block
    outputs = dense(attention_output)

    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    dense_units,
    dropout_rate,
):

    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))
    outputs = inputs

    # iteratively generate transformer block
    for _ in range(num_transformer_blocks):

        # normalise
        norm_one = layers.LayerNormalization(epsilon=1e-6)(outputs)

        # multi-head self-attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(norm_one, norm_one)

        # first skip connection
        skip_one = layers.Add()([attention_output, outputs])

        # normalise again and apply dense network
        norm_two = layers.LayerNormalization(epsilon=1e-6)(skip_one)
        dense = dense_network(dense_units, dropout_rate)(norm_two)

        # second skip connection
        outputs = layers.Add()([dense, skip_one])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class Perceiver(keras.Model):
    """
    Class representing full model of the perceiver transformer.
    """

    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        dense_units,
        dropout_rate,
        num_iterations,
        classifier_units,
    ):
        super(Perceiver, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):

        # latent array
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # patching module.
        self.patcher = Patches(self.patch_size)

        # patch encoder.
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)

        # cross-attention module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.dense_units,
            self.dropout_rate,
        )

        # transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.dense_units,
            self.dropout_rate,
        )

        # global average pooling
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # dense classification block
        self.classification_head = dense_network(self.classifier_units, self.dropout_rate)

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):

        # Create patches.
        patches = self.patcher(inputs)

        # Encode patches.
        encoded_patches = self.patch_encoder(patches)

        # Prepare cross-attention inputs.
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }

        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):

            # Apply cross-attention from the latent array to the data array.
            latent_array = self.cross_attention(cross_attention_inputs)

            # Apply self-attention Transformer to the latent array.
            latent_array = self.transformer(latent_array)

            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # Apply global average pooling to generate a [batch_size, projection_dim] representation tensor.
        representation = self.global_average_pooling(latent_array)

        # Generate logits.
        logits = self.classification_head(representation)
        return logits


