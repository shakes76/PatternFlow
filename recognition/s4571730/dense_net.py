from tensorflow.keras import layers
import tensorflow as tf 


def dense_block(dense_layers):
    block = tf.keras.Sequential()
    # for units in dense_layers[:-1]:
        # use GELU activation as in paper
    block.add(layers.LayerNormalization())
    block.add(layers.Dense(dense_layers[0], activation=tf.nn.gelu))

        # Final linear layer
    block.add(layers.Dense(units=dense_layers[-1]))
    # block.add(layers.Dropout(dropout_rate))
    return block