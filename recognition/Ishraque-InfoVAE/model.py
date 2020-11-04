import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, losses
import tensorflow_probability as tfp
from data import img_size

def get_encoder(latent_dim):
    input_layer = layers.Input(shape=(img_size, img_size, 1))
    c1 = layers.Conv2D(filters=32, kernel_size=3, input_shape=(2, 2), activation='relu')(input_layer)
    c2 = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(c1)
    d1 = layers.Flatten()(c2)
    d2 = layers.Dense(latent_dim)(d1)
    return models.Model(inputs=input_layer, outputs=d2)

def get_decoder(latent_dim):
    input_layer = layers.Input(shape=(latent_dim, )) # Ensure this matches output of encoder network
    d1 = layers.Dense((img_size//4)*(img_size//4)*32, activation='relu')(input_layer)
    r1 = layers.Reshape(target_shape=((img_size//4), (img_size//4), 32))(d1)
    c1 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(r1)
    c2 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(c1)
    c3 = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(c2)
    return models.Model(inputs=input_layer, outputs=c3)

class InfoVAE():
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.encoder = get_encoder(latent_dim)
        self.decoder = get_decoder(latent_dim)
        i = keras.Input(shape=(img_size, img_size, 1))
        e = self.encoder(i)
        x = self.decoder(e)
        self.model = keras.Model(inputs=i, outputs=x)
        self.optimizer = optimizers.Adam()

    def _compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    @tf.function
    def _encoder_loss(self, latent_encoding: tf.Tensor):
        # actual_dist = tf.random.normal(shape=(batch_size, self.latent_dim))
        actual_dist = tf.random.normal(shape=latent_encoding.shape)
        return self._compute_mmd(actual_dist, latent_encoding)

    def _dssim_loss_scalar(self, shape, images):
        actual_dist = tf.random.normal(shape=shape)
        reconstruction = self.decoder(actual_dist, training=True)
        # Compute SSIM over tf.float32 Tensors.
        ssim2 = tf.image.ssim(images, reconstruction, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
        return (1-tf.reduce_mean(ssim2, axis=None))/2

    @tf.function
    def train(self, images: tf.Tensor):
        loss = -1
        with tf.GradientTape() as tape:
            latent_encoding = self.encoder(images, training=True)
            reconstruction = self.decoder(latent_encoding, training=True)
            enc_loss = self._encoder_loss(latent_encoding)
            rec_loss = losses.mean_squared_error(images, reconstruction)

            # SSIM loss
            dssim = self._dssim_loss_scalar(latent_encoding.shape, images)
            loss = (enc_loss + rec_loss)*(1+dssim)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return tf.reduce_mean(loss, axis=None)
    
    @tf.function
    def get_loss(self, images: tf.Tensor):
        """Get loss for a given validation set"""
        latent_encoding = self.encoder(images, training=False)
        reconstruction = self.decoder(latent_encoding, training=False)
        enc_loss = self._encoder_loss(latent_encoding)
        rec_loss = losses.mean_squared_error(images, reconstruction)

        return tf.math.reduce_mean(enc_loss + rec_loss, axis=None)

    @tf.function
    def random_reconstruction_sample(self, n):
        """Returns n reconstructed images"""
        latent_encoding = tf.random.normal(shape=(n, self.latent_dim))
        return self.decoder(latent_encoding, training=False)