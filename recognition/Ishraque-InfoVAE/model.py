import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import tensorflow_probability as tfp
from data import img_size

def get_encoder(latent_dim):
    input_layer = layers.Input(shape=(img_size, img_size, 1))
    c1 = layers.Conv2D(filters=32, kernel_size=3, input_shape=(2, 2), activation='relu')(input_layer)
    c2 = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(c1)
    d1 = layers.Flatten()(c2)
    d2 = layers.Dense(latent_dim)
    return models.Model(input=input_layer, output=d2)

def get_decoder(latent_dim):
    input_layer = layers.Input(shape=(latent_dim, )) # Ensure this matches output of encoder network
    d1 = layers.Dense((img_size//4)*(img_size//4)*32, activation='relu')(input_layer)
    r1 = layers.Reshape(target_shape=((img_size//4), (img_size//4), 32))(d1)
    c1 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(r1)
    c2 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(c1)
    c3 = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(c2)
    return models.Model(input=input_layer, output=c3)

class InfoVAE():
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(InfoVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = get_encoder(latent_dim)
        self.decoder = get_decoder(latent_dim)
        x = self.decoder(self.encoder)
        self.model = models.Model(input=self.encoder, output=self.decoder)
        self.optimizer = optimizers.Adam()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), 2, 1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits