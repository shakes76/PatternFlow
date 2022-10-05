import tensorflow as tf
import numpy as np

class AE(tf.keras.Model):
    def __init__(self, latent_dim=4, image_shape=(28,28,1), **kwargs):
        super(AE, self).__init__(**kwargs)

        self.latent_dim = latent_dim

        input = tf.keras.layers.Input(shape=image_shape, name="input")
        x = tf.keras.layers.Conv2D(
            filters = 32, 
            kernel_size = 3, 
            strides = 2, 
            activation = 'relu',
            padding = "same", 
            name = "compression_1")(input)
        x = tf.keras.layers.Conv2D(
            filters = 64, 
            kernel_size=3, 
            strides=2, 
            activation='relu',
            padding = "same", 
            name = "compression_2")(x)
        x = tf.keras.layers.Conv2D(
            filters = 128, 
            kernel_size=3, 
            strides=2, 
            activation='relu',
            padding = "same", 
            name = "compression_3")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        self.latent_space = tf.keras.layers.Dense(latent_dim, name="latent_space")(x)

        self.encoder = tf.keras.Model(input, self.latent_space, name="encoder")

        input = tf.keras.layers.Input(shape=latent_dim, name="input")
        x = tf.keras.layers.Dense(
            np.prod([max(int(np.floor(el/(2*2*2))),1) for el in image_shape]), 
            activation = 'relu', 
            name = "expand")(input)
        x = tf.keras.layers.Reshape(
            target_shape = (max(int(np.floor(el/(2*2*2))),1) for el in image_shape), 
            name = "reshape")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 128, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            activation = 'relu', 
            name = "reconstruct_1")(x)
        x = tf.keras.layers.ZeroPadding2D(((0,1),(0,1)))(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 64, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            activation = 'relu', 
            name = "reconstruct_2")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 32, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            name = "to_image", 
            activation = "sigmoid")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 3,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            activation = 'sigmoid')(x)

        self.decoder = tf.keras.Model(input, x, name="decoder")

    def train_step(self, train_data):
        x, _ = train_data
        with tf.GradientTape() as tape:
            out = self.decoder(self.encoder(x))
            loss = tf.keras.losses.mean_squared_error(x, tf.reshape(out, shape=tf.shape(x)))
        
        gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return { "loss": loss }

    def test_step(self, test_data):
        x, _ = test_data
        out = self.decoder(self.encoder(x))
        loss = tf.keras.losses.mean_squared_error(x, tf.reshape(out, shape=tf.shape(y)))

        return { "loss": loss }

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))
