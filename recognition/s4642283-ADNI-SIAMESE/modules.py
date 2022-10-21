import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras import layers
from keras.layers import Lambda

def siamese_model(input_shape):
    """
    Siamese model with distance computation and final sigmoid layer.
    """

    # CNN and Pooling layers
    input = layers.Input(input_shape)
    x = layers.Conv2D(32, (10, 10), activation="relu")(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (7, 7), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (4, 4), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (4, 4), activation="relu")(x)
    x = layers.Flatten()(x)

    # Produce 10 dimensional vector
    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dense(10, activation="relu")(x)
    embedding_network = keras.Model(input, x)

    # Inputs for Siamese Networks
    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)

    # Both Siamese Networks have the same embedding network
    network_1 = embedding_network(input_1)
    network_2 = embedding_network(input_2)

    # Compute L1 Distance between two embeddings
    L1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    distance_computation_layer = L1_distance_layer([network_1, network_2])
    normal_layer = tf.keras.layers.BatchNormalization()(distance_computation_layer)

    # Classify whether the images belong to the same class or different classes
    prediction = layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=prediction)

    return siamese