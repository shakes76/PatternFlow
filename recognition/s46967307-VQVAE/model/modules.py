import tensorflow as tf
import numpy as np

latent_dims = 100
image_shape = (256,256,1)
num_embedings = 100

class VQ(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embeddings, latent_dim, **kwargs):
        super(VQ, self).__init(**kwargs)
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        
        w_init = tf.random_normal_initializer()
        self.embeddings = tf.Variable(
                trainable=True,
                name="embeddings",
                initial_value=w_init(shape=(self.num_embeddings, self.latent_dim), dtype="float32")
                )

    def call(self, inputs):
        # Each result is a vector of distances from associated input and each embedding
        result = tf.vectorized_map(lambda x:
                tf.vectorized_map(lambda y:
                    tf.norm(tf.math.subtract(y, x)),
                    self.embeddings),
                inputs)
        return result

class AE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AE, self).__init__(**kwargs)

        # ------ ENCODER -------
        # Takes image as input, runs it through 3 convolutional
        # layers which each halve the size of the image.
        # The remaining image is flattened and shrunk into a 
        # latent space.
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
        self.latent_space = tf.keras.layers.Dense(latent_dims, name="latent_space")(x)

        self.encoder = tf.keras.Model(input, self.latent_space, name="encoder")

        # ------ VQ Layer ------
        # Takes output from encoder.
        # Returns the closest vector in the embedding to the latent
        # space.
        input = tf.keras.layers.Input(shape=latent_dim, name="input")
        x = VQ(self.num_embeddings, self.embeddings, self.latent_dim)(input)
        self.vq = tf.keras.Model(input, x, name="vq")

        # ------ DECODER -------
        # Takes output from VQ layer. 
        # Structure is identical to encoder but with Conv2DTranspose
        # to upscale the image rather than downscale.
        input = tf.keras.layers.Input(shape=latent_dims, name="input")
        x = tf.keras.layers.Dense(
            32*32*32,
            activation = 'relu', 
            name = "expand")(input)
        x = tf.keras.layers.Reshape(
            target_shape = (32,32,32),
            name = "reshape")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 64, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            activation = 'relu', 
            name = "reconstruct_1")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 128, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            activation = 'relu', 
            name = "reconstruct_2")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 64, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            name = "reconstruct_3", 
            activation = "sigmoid")(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters = 1,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            name = "to_image",
            activation = 'sigmoid')(x)

        self.decoder = tf.keras.Model(input, x, name="decoder")

    def train_step(self, train_data):
        x, _ = train_data
        with tf.GradientTape() as tape:
            out = self.call(x)
            # MSE loss is taken on original image and encoded(decoded) image
            loss = tf.keras.losses.mean_squared_error(x, tf.reshape(out, shape=tf.shape(x)))
        
        gradients = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return { "loss": loss }

    def test_step(self, test_data):
        x, _ = test_data
        out = self.call(x)
        loss = tf.keras.losses.mean_squared_error(x, tf.reshape(out, shape=tf.shape(y)))

        return { "loss": loss }

    def call(self, inputs):
        return self.decoder(self.vq(self.encoder(inputs)))
