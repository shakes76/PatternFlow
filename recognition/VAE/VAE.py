import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras import Sequential


class VAENetwork(tf.keras.Model):
    def __init__(self, latent_dimension, kernel_size=3, strides=2):
        super(VAENetwork, self).__init__()
        # number of dimensions of the latent distribution
        self.latent_dim = latent_dimension
        # the encoder
        self.encoder = self.define_encoder(latent_dimension, kernel_size, strides)
        # the decoder
        self.decoder = self.define_decoder(latent_dimension, kernel_size, strides)

    @tf.function
    # decode the points sampled from the latent distribution
    def sample_z(self, z=None):
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z, sigmoid=True)

    def encode(self, x):
        parameters = self.encoder(x)
        mean, log_var = tf.split(parameters, num_or_size_splits=2, axis=1)
        return mean, log_var

    # implement the reparameterize trick
    def reparameterize(self, mean, log_var):
        # generate epsilon from a standard normal distribution
        epsilon = tf.random.normal(shape=mean.shape)
        return epsilon * tf.exp(log_var / 2) + mean

    # decode a latent representation
    def decode(self, z, sigmoid=False):
        logits = self.decoder(z)
        if sigmoid:  # apply signmoid function
            probability = tf.sigmoid(logits)
            return probability
        return logits

    # generate images by encoding and decoding the test samples
    def generate_images(self, model, test_sample):
        for x_test in test_sample:
            # encode the test samples and get the parameters of the latent distribution
            mean, log_var = model.encode(tf.expand_dims(x_test, axis=-1))
            # reparameter trick
            z = model.reparameterize(mean, log_var)
            # decode the points sampled from the latent distribution to generate images
            predictions = model.sample_z(z)
            return predictions

    # defining the encoder
    def define_encoder(self, latent_dimension, kernel_size, strides):
        e = Sequential()
        e.add(InputLayer(input_shape=(256, 256, 1)))
        # downsample the side length by 0.5
        e.add(Conv2D(filters=16, kernel_size=kernel_size, strides=strides, activation='relu'))
        e.add(BatchNormalization()) # BatchNormalization layer to improve the performance
        # downsample the side length by 0.5
        e.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation='relu'))
        e.add(BatchNormalization())
        # downsample the side length by 0.5
        e.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, activation='relu'))
        e.add(BatchNormalization())
        # flatten the image pixels
        e.add(Flatten())
        # compress the image pixels to 2 * latent_dimension
        e.add(Dense(latent_dimension + latent_dimension))
        return e

    # defining the decoder
    def define_decoder(self, latent_dimension, kernel_size, strides):
        d = Sequential()
        d.add(InputLayer(input_shape=(latent_dimension,)))
        d.add(Dense(units=32*32*32, activation='relu'))
        d.add(BatchNormalization())
        d.add(Reshape(target_shape=(32, 32, 32)))
        # upsample the side length by 2
        d.add(Conv2DTranspose(filters=64, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        d.add(BatchNormalization())
        # upsample the side length by 2
        d.add(Conv2DTranspose(filters=32, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        d.add(BatchNormalization())
        # upsample the side length by 2
        d.add(Conv2DTranspose(filters=16, kernel_size=kernel_size, strides=strides, padding='same', activation='relu'))
        d.add(BatchNormalization())
        # no activation
        d.add(Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=1, padding='same'))
        return d