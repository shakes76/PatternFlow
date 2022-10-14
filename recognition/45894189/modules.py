from turtle import xcor
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

class Noise(layers.Layer):
    """
    Noise input layer [B]
    """
    def build(self):
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(shape = [1, 1, 1, 1], initializer=initializer, trainable=True)

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output

class AdaIN(layers.Layer):
    """
    Combined AdaIN and [A] layer. Affine transform of W is used to perform
    Instance Normalisation on X values 
    """
    def __init__(self, epsilon=1e-8):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def build(self):
        self.dense_ys = layers.Dense(1)
        self.dense_yb = layers.Dense(1)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_ys(w), (-1, 1, 1, 1))
        yb = tf.reshape(self.dense_yb(w), (-1, 1, 1, 1))
        axes = list(range(1, len(x.shape) - 1))
        mean = tf.math.reduce_mean(x, axes, keepdims=True)
        stdev = tf.math.reduce_std(x, axes, keepdims=True) 
        return ys * ((x - mean)/(stdev + self.epsilon)) + yb

def WNetwork(latent_dim=256):
    """
    Mapping Network z -> w mapping latent noise to style code with set of
    fully connected layers
    returns: Mapping Network
    """
    z = layers.Input(shape=[latent_dim])
    w = z
    for _ in range(8):
        w = layers.Dense(latent_dim)(z)
        w = layers.LeakyReLU(0.2)(z)
    return tf.keras.Model(z, w)

def discriminator():
    """
    Discriminator model, takes in 256x256x1 images and calssifies them as real or fake. Built with initial input and convolution layer,
    then repeated discriminator blocks until the image is downsampled to 4x4, and finished with 2 convolutions layers, then a flatten and
    Dense classification layer
    """
    current_size = 256
    n_filters = 16
    input = layers.Input(shape=[current_size, current_size, 1])
    x = input
    x = layers.Conv2D(n_filters, kernel_size=1, padding="valid", input_shape=(current_size, current_size, 1))(x)
    while current_size > 4:
        x = discriminator_block(x, n_filters, current_size)
        current_size = current_size // 2
        n_filters = 2 * n_filters
    x = layers.Conv2D(n_filters, kernel_size=3, padding="valid", input_shape=(current_size, current_size, n_filters))(x)
    x = layers.LeakyReLU(alpha=0.2)
    x = layers.Conv2D(n_filters, kernel_size=4, padding="valid", input_shape=(current_size, current_size, n_filters))(x)
    x = layers.LeakyReLU(alpha=0.2)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="linear")
    return tf.keras.Model(inputs=[input], outputs=x)

def discriminator_block(input_tensor, n_filters, image_size):
    """
    Main block for discriminator containing two convolutional layers with LeakyReLU activation. the second layer doubles the number
    of filters. The block is ended with a downsampling of the image that halves it's size
    """
    x = layers.Conv2D(n_filters, kernel_size=3, padding="valid", input_shape=(image_size, image_size, n_filters))(input_tensor)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(2*n_filters, kernel_size=3, padding="valid", input_shape=(image_size, image_size, n_filters))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Resizing(image_size // 2, image_size // 2)(x)
    return x
