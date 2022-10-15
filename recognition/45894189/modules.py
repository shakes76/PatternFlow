import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

class Noise(layers.Layer):
    """
    Noise input layer [B]
    """
    def build(self, input_shape):
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(shape=[1, 1, 1, 1], initializer=initializer, trainable=True)

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

    def build(self, input_shape):
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

class Generator():
    def __init__(self):
        self.init_size = 4
        self.init_filters = 256
        self.noise_inputs = [tf.random.normal([32, res, res, 1]) for res in [4, 8, 16, 32, 64, 128, 256]]

    def generator_block(self, n_filters, image_size):
        input_tensor = layers.Input(shape=(image_size, image_size, n_filters))
        noise = layers.Input(shape=(2*image_size, 2*image_size, 1))
        w = layers.Input(shape=256)

        x = input_tensor
        n_filters = n_filters // 2

        x = layers.UpSampling2D((2,2))(x)
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = Noise()([x, noise])
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = AdaIN()([x, w])

        x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
        x = Noise()([x, noise])
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = AdaIN()([x, w])

        return tf.keras.Model([input_tensor, w, noise], x)

    def generator(self):
        current_size = self.init_size
        n_filters = self.init_filters
        input = layers.Input(shape=(current_size, current_size, n_filters))

        noise_inputs = []
        curr_size = self.init_size
        while curr_size <= 256:
            noise_inputs.append(layers.Input(shape=[curr_size, curr_size, 1]))
            curr_size *= 2

        w = layers.Input(shape=n_filters)
        x = input
        i = 0

        x = Noise()([x, self.noise_inputs[i]])
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = AdaIN()([x, w])
        x = layers.Conv2D(256, kernel_size=3, padding="same")(x)

        x = Noise()([x, self.noise_inputs[i]])
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = AdaIN()([x, w])

        while current_size < 256:
            i += 1
            x = self.generator_block(n_filters, current_size)([x, w, noise_inputs[i]])
            current_size = 2 * current_size
            n_filters = n_filters // 2

        x = layers.Conv2D(1, kernel_size=4, padding="same", activation="tanh")(x)
        return tf.keras.Model([input, w, noise_inputs], x)

class Discriminator():

    def __init__(self):
        self.init_size = 256
        self.init_filters = 16

    def discriminator_block(self, n_filters, image_size):
        """
        Main block for discriminator containing two convolutional layers with LeakyReLU activation. the second layer doubles the number
        of filters. The block is ended with a downsampling of the image that halves it's size
        """
        input_tensor = layers.Input(shape=(image_size, image_size, n_filters))
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same", input_shape=(image_size, image_size, n_filters))(input_tensor)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(2*n_filters, kernel_size=3, padding="same", input_shape=(image_size, image_size, n_filters))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.AveragePooling2D((2,2))(x)   # Downsizing

        return tf.keras.Model(input_tensor, x)

    def discriminator(self):
        """
        Discriminator model, takes in 256x256x1 images and calssifies them as real or fake. Built with initial input and convolution layer,
        then repeated discriminator blocks until the image is downsampled to 4x4, and finished with 2 convolutions layers, then a flatten and
        Dense classification layer
        """
        current_size = self.init_size
        n_filters = self.init_filters
        input_tensor = layers.Input(shape=[current_size, current_size, 1])
        x = input_tensor

        x = layers.Conv2D(n_filters, kernel_size=1, padding="same", input_shape=(current_size, current_size, 1))(x)

        while current_size > 4:
            x = self.discriminator_block(n_filters, current_size)(x)
            current_size = current_size // 2
            n_filters = 2 * n_filters

        x = layers.Conv2D(n_filters, kernel_size=3, padding="same", input_shape=(current_size, current_size, n_filters))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(n_filters, kernel_size=4, padding="same", input_shape=(current_size, current_size, n_filters))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(1, activation="linear")(x)

        return tf.keras.Model(input_tensor, x)
