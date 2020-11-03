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

    def _encoder_loss(self, latent_encoding: tf.Tensor):
        # actual_dist = tf.random.normal(shape=(batch_size, self.latent_dim))
        actual_dist = tf.random.normal(shape=latent_encoding.shape)
        return self._compute_mmd(actual_dist, latent_encoding)

    @tf.function
    def train(self, images: tf.Tensor):
        loss = -1
        with tf.GradientTape() as tape:
            latent_encoding = self.encoder(images, training=True)
            reconstruction = self.decoder(latent_encoding)
            enc_loss = self._encoder_loss(latent_encoding)
            rec_loss = losses.mean_squared_error(images, reconstruction)
            loss = enc_loss + rec_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean = tf.split(self.encoder(x), 2, 1)
        return mean

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        # if apply_sigmoid:
        #     probs = tf.sigmoid(logits)
        #     return probs
        return logits