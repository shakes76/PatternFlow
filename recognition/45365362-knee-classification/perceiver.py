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
    def __init__(self, projection_dim):
        super(PatchEncoder, self).__init__()
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=PATCH_COUNT, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=PATCH_COUNT, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded