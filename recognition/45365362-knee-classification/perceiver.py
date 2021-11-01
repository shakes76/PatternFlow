from process_data import process_data
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow_addons as tfa

num_classes = 2
input_shape = (228, 260, 3)

X_train, y_train, X_test, y_test = process_data("AKOA_Analysis\AKOA_Analysis", 80, 20)

PATCH_SIZE = 2
PATCH_COUNT = (128 // PATCH_SIZE) ** 2
PROJECTION_DIMENSION = 256
LATENT_DIMENSIONS = 256
ffn_units = [
    PROJECTION_DIMENSION,
    PROJECTION_DIMENSION,
]
HEAD_COUNT = 8
TRANSFORMER_BLOCK_COUNT = 4

# feed forward
def get_feed_forward_network(hidden_units, dropout_rate):
    
    network_layers = []
    for units in hidden_units[:-1]:
        network_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    network_layers.append(layers.Dense(units=hidden_units[-1]))
    network_layers.append(layers.Dropout(dropout_rate))

    network = keras.Sequential(network_layers)
    return network


class Patches(layers.Layer):
    def __init__(self, ):
        super(Patches, self).__init__()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        self.projection = layers.Dense(units=PROJECTION_DIMENSION)
        self.position_embedding = layers.Embedding(
            input_dim=PATCH_COUNT, output_dim=PROJECTION_DIMENSION
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=PATCH_COUNT, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


def get_cross_attention(
    data_dim, ffn_units, dropout_rate
):

    inputs = {
        "latent_array": layers.Input(shape=(LATENT_DIMENSIONS, PROJECTION_DIMENSION)),
        "data_array": layers.Input(shape=(data_dim, PROJECTION_DIMENSION)),
    }


    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    query = layers.Dense(units=PROJECTION_DIMENSION)(latent_array)
    key = layers.Dense(units=PROJECTION_DIMENSION)(data_array)
    value = layers.Dense(units=PROJECTION_DIMENSION)(data_array)

    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )

    attention_output = layers.Add()([attention_output, latent_array])

    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    feed_forward_network = get_feed_forward_network(hidden_units=ffn_units, dropout_rate=dropout_rate)
   
    outputs = feed_forward_network(attention_output)
    outputs = layers.Add()([outputs, attention_output])

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_transformer(
    ffn_units,
    dropout_rate,
):

    inputs = layers.Input(shape=(LATENT_DIMENSIONS, PROJECTION_DIMENSION))

    x0 = inputs
    for _ in range(TRANSFORMER_BLOCK_COUNT):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        attention_output = layers.MultiHeadAttention(
            num_heads=HEAD_COUNT, key_dim=PROJECTION_DIMENSION, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, x0])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn = get_feed_forward_network(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        x0 = layers.Add()([x3, x2])

    model = keras.Model(inputs=inputs, outputs=x0)
    return model