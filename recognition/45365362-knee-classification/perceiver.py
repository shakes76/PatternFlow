from process_data import process_data
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow_addons as tfa

num_classes = 2
input_shape = (228, 260, 3)

X_train, y_train, X_test, y_test = process_data("AKOA_Analysis\AKOA_Analysis", 80, 20)

PATCH_SIZE = 2

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