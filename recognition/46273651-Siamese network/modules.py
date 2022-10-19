import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


embedding_model = tf.keras.Sequential(
    [
        keras.Input(shape=(224, 224, 1)),
        layers.Conv2D(64, (10,10), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (7,7), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (4,4), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (4,4), activation="relu"),

        layers.Flatten(),
        layers.Dense(512, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
    ],
    name="embedding"
)


# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


input_1 = layers.Input(name="pair_x", shape=(224, 224, 1))
input_2 =layers.Input(name="pair_y", shape=(224, 224, 1))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_model(input_1)
tower_2 = embedding_model(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)


def main():
    embedding_model.summary()
    siamese.summary()

if __name__ == '__main__':
    main()