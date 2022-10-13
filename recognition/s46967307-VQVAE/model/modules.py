import tensorflow as tf
import numpy as np

latent_dims = 8
image_shape = (256, 256, 1)
num_embeddings = 32
beta = 2.0

class PixelConvLayer(tf.keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = tf.keras.layers.Conv2D(**kwargs)
    
    def build(self, input_shape):
        self.conv.build(input_shape)

        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[:kernel_shape[0] // 2, ...] = 1.0 # Sets points above center to 1
        self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0 # Sets points left of center to 0

        # Pixelcnn uses 3 channels on mask B, but we only have grayscale data
        # Only change is to include centre pixel in mask
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
    
    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

def get_pixel_cnn(kernel_size, input_shape):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float64)
    #onehot = tf.one_hot(inputs, num_embeddings)
    x = PixelConvLayer(
        mask_type="A",
        filters=128,
        kernel_size=kernel_size,
        activation="relu",
        padding="same")(inputs)

    for _ in range(8):
        x = PixelConvLayer(
            mask_type="B",
            filters=32,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid")(x)
        
        x = tf.keras.layers.BatchNormalization()(x)

    # Flatten each pixel down to the number of embeddings
    x= tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="valid")(x)
    
    return tf.keras.Model(inputs, x)

def get_indices(embeddings, inputs_flat):
    results = tf.vectorized_map(
        lambda y:
        tf.vectorized_map(
            lambda x:
            tf.norm(tf.math.subtract(x, y)),
            embeddings),
        inputs_flat)

    results = tf.math.argmin(results, axis=1)
    results = tf.matmul(tf.one_hot(
        results, num_embeddings), embeddings)

    return results
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

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        inputs_flat = tf.reshape(inputs, shape=(-1, latent_dims))

        results = get_indices(self.embeddings, inputs_flat)

        codebook_loss = tf.reduce_mean(
            tf.square(tf.stop_gradient(results) - inputs_flat))
        commitment_loss = tf.reduce_mean(
            tf.square(results - tf.stop_gradient(inputs_flat))) * beta
        self.add_loss(commitment_loss + codebook_loss)

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
