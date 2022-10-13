import tensorflow as tf
from tensorflow import keras
from keras import layers

class Noise(layers.Layer):
    def build(self):
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(shape = [1, 1, 1, 1], initializer=initializer, trainable=True)

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output

class AdaIN(layers.Layer):
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
    z = layers.Input(shape=[latent_dim])
    w = z
    for _ in range(8):
        w = layers.Dense(256)(z)
        w = layers.LeakyReLU(0.2)(z)
    return tf.keras.Model(z, w)