import tensorflow as tf
import numpy as np

latent_dims = 8
image_shape = (256, 256, 1)
num_embeddings = 32
beta = 2.0


class GAN(tf.keras.Model):
    def __init__(self, vq, **kwargs):
        super(GAN, self).__init__(**kwargs)

        self.vq = vq

        # Define generator model, should map random 32x32x1 noise to 32x32x1 indexes into
        # the vqvae's embeddings in range [0..=embeddings]
        input = tf.keras.layers.Input(
            shape=(32, 32, 1), batch_size=None, name="input")
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            activation='relu',
            padding="same",
            name="noise1")(input)
        x = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=6,
            strides=1,
            activation=tf.keras.layers.ReLU(max_value=num_embeddings),
            padding="same",
            name="to_latent")(x)
        x = tf.keras.layers.Lambda(
           lambda y: tf.math.round(y)
        )(x)

        self.generator = tf.keras.Model(input, x, name="encoder")

        # Define the discriminator model, should take a 256x256x1 image, do some shrinking
        # convolution, and then dense layer to a single logit output from 0 to 1
        input = tf.keras.layers.Input(
            shape=image_shape, batch_size=None, name="input")
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            activation='relu',
            padding="same",
            name="conv1")(input)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation='softmax')(x)

        self.discriminator = tf.keras.Model(input, x, name="discriminator")

    def train_step(self, train_data):
        # x is images of real brains (256,256,1)
        x, _ = train_data

        # y is random noise of size (len(x), 32, 32, 1)
        y = tf.random.uniform((tf.shape(x)[0], 32, 32, 1))

        # Generate Images
        g = tf.cast(tf.math.round(self.generator(y)),
                    dtype=tf.int64)  # Generated images
        g = tf.reshape(g, shape=(8, 32, 32))
        g = tf.gather(self.vq.get_layer(name="vq").variables[0], g)
        g = self.vq.decoder(g)

        with tf.GradientTape() as tape:
            # Discriminate
            rd = self.discriminator(x)  # Discriminator real inputs
            fd = self.discriminator(g)  # Discriminator fake inputs

            dl = tf.keras.losses.BinaryCrossentropy()(
                tf.concat([tf.ones_like(rd),tf.zeros_like(fd)], axis=0), 
                tf.concat([rd, fd], axis=0)
            )

        d_grad = tape.gradient(dl, self.discriminator.trainable_variables)

        self.optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # y is random noise of size (len(x), 32, 32, 1)
        y = tf.random.uniform((tf.shape(x)[0], 32, 32, 1))

        # Generate Images
        g = tf.cast(tf.math.round(self.generator(y)), dtype=tf.int64)
        g = tf.reshape(g, shape=(8, 32, 32))
        g = tf.gather(self.vq.get_layer(name="vq").variables[0], g)
        g = self.vq.decoder(g)
        
        fd = self.discriminator(g)

        with tf.GradientTape() as tape:
            gl = self.generator(y) - tf.stop_gradient(self.generator(y) - tf.keras.losses.BinaryCrossentropy()(
                fd,
                tf.zeros_like(fd)
            ))
        
        g_grad = tape.gradient(gl, self.generator.trainable_variables)

        self.optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return {"discriminator_loss": dl, "generator_loss": gl }

class VQ(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VQ, self).__init__(**kwargs)

        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            trainable=True,
            name="embeddings",
            initial_value=w_init(
                shape=(num_embeddings, latent_dims), dtype="float32")
        )

    def get_indices(self, inputs_flat):
        results = tf.vectorized_map(
            lambda y:
            tf.vectorized_map(
                lambda x:
                tf.norm(tf.math.subtract(x, y)),
                self.embeddings),
            inputs_flat)

        results = tf.math.argmin(results, axis=1)
        results = tf.matmul(tf.one_hot(
            results, num_embeddings), self.embeddings)

        codebook_loss = tf.reduce_mean(
            tf.square(tf.stop_gradient(results) - inputs_flat))
        commitment_loss = tf.reduce_mean(
            tf.square(results - tf.stop_gradient(inputs_flat))) * beta
        self.add_loss(commitment_loss + codebook_loss)

        return results

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        inputs_flat = tf.reshape(inputs, shape=(-1, latent_dims))

        results = self.get_indices(inputs_flat)

        # Reshape results back into compressed image
        results = tf.reshape(results, shape=inputs_shape)

        return inputs + tf.stop_gradient(results - inputs)

class AE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AE, self).__init__(**kwargs)

        # ------ ENCODER -------
        # Takes image as input, runs it through 3 convolutional
        # layers which each halve the size of the image.
        # The remaining image is flattened and shrunk into a
        # latent space.
        input = tf.keras.layers.Input(shape=image_shape, batch_size=None, name="input")
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
        x = tf.keras.layers.Conv2D(
            filters = latent_dims, 
            kernel_size=3,
            strides=1,
            activation='relu',
            padding = "same", 
            name = "to_latent")(x)

        self.encoder = tf.keras.Model(input, x, name="encoder")

        # ------ VQ Layer ------
        # Takes output from encoder.
        # Returns the closest vector in the embedding to the latent
        # space.
        input = tf.keras.layers.Input(shape=(32,32,latent_dims), batch_size=None, name="input")
        x = VQ(name="vq")(input)
        self.vq = tf.keras.Model(input, x, name="vq")

        # ------ DECODER -------
        # Takes output from VQ layer.
        # Structure is identical to encoder but with Conv2DTranspose
        # to upscale the image rather than downscale.
        input = tf.keras.layers.Input(shape=(32,32,latent_dims), batch_size=None, name="input")
        x = tf.keras.layers.Conv2DTranspose(
            filters = 128, 
            kernel_size = 3, 
            strides = 2, 
            padding = 'same',
            activation = 'relu', 
            name = "reconstruct_1")(input)
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

            rc_loss = tf.keras.losses.mean_squared_error(
                x, tf.reshape(out, shape=tf.shape(x)))
            vq_loss = sum(self.vq.losses)

            loss = rc_loss + vq_loss

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, test_data):
        x, _ = test_data
        out = self.call(x)
        rc_loss = tf.keras.losses.mean_squared_error(
            x, tf.reshape(out, shape=tf.shape(x)))
        vq_loss = sum(self.vq.losses)

        loss = rc_loss + vq_loss

        return {"loss": loss}

    def call(self, inputs):
        return self.decoder(self.vq(self.encoder(inputs)))
