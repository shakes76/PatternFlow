import tensorflow as tf


# Mapping Network
class G_Mapping:

    def __init__(self, latent_size):
        self.latent_size = latent_size
        # Mapping network for generator
        self.model = self.mapping_nw()

    # build the mapping network
    def mapping_nw(self):
        fc_input = tf.keras.layers.Input(shape=(self.latent_size,))
        fc1 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc_input)
        fc2 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc1)
        fc3 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc2)
        fc4 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc3)
        fc5 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc4)
        fc6 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc5)
        fc7 = tf.keras.layers.Dense(self.latent_size, activation="relu")(fc6)
        return tf.keras.Model(inputs=[fc_input], outputs=[fc7])


